import numpy as np
import math
from scipy.signal import butter, sosfilt

class STIAnalyzer:
    """Speech Transmission Index analyzer"""

    def __init__(self, audio, sr):
        # Mirror the CLI path: mono + float64 to avoid dtype/channel drift
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.audio = np.asarray(audio, dtype=np.float64)
        self.sr = int(sr)

        # STI parameters (octave bands + weights)
        self.band_centers = [125, 250, 500, 1000, 2000, 4000, 8000]
        self.band_weights = [0.01, 0.04, 0.146, 0.212, 0.308, 0.244, 0.04]

        # Standard modulation frequencies (14 values, 0.63 to 12.5 Hz)
        self.mod_freqs = np.array(
            [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5],
            dtype=np.float64
        )

    def design_octave_band(self, center_freq):
        """
        Designs a 4th-order Butterworth bandpass filter for one-octave band 
        around 'center_freq'. Boundaries: cf/sqrt(2) .. cf*sqrt(2).
        """
        low = center_freq / math.sqrt(2)
        high = center_freq * math.sqrt(2)
        nyq = 0.5 * self.sr
        low_cut = max(low / nyq, 1e-5)
        high_cut = min(high / nyq, 0.99999)
        if low_cut >= 1:
            return None
        if high_cut >= 1:
            high_cut = 0.99999
        sos = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', output='sos')
        return sos

    def compute_sti(self, window_dur=0.5, hop_dur=0.25):
        """
        Compute STI with the exact same logic as the standalone script's compute_sti.
        Returns a dict (keeping your richer API) whose 'time_stamps' and 'sti_values'
        match the script's outputs on the same audio/fs/window/hop.
        """
        fs = self.sr
        audio = self.audio
        nyquist = 0.5 * fs

        # ---- Valid bands + weight renorm (identical) ----
        valid_bands = []
        valid_weights = []
        for center, w in zip(self.band_centers, self.band_weights):
            if center / math.sqrt(2) < nyquist * 0.999:
                valid_bands.append(center)
                valid_weights.append(w)
        valid_weights = np.array(valid_weights, dtype=np.float64)
        if valid_weights.size == 0:
            # No valid bands -> return zeros in your expected structure
            return {
                'overall_sti': 0.0, 'sti_min': 0.0, 'sti_max': 0.0, 'sti_std': 0.0,
                'temporal_sti': [], 'time_stamps': np.array([]), 'sti_values': np.array([]),
                'band_sti': {}, 'band_centers': [], 'band_weights': []
            }
        valid_weights /= valid_weights.sum()

        # ---- Filter per band (same ordering as script: zip(valid_bands, valid_weights)) ----
        band_signals = {}
        for center, _Wk in zip(valid_bands, valid_weights):
            sos = self.design_octave_band(center)
            if sos is None:
                # In practice shouldn't happen with the valid_bands test,
                # but if it does, skip like the script implicitly assumes it won't.
                continue
            band_signals[center] = sosfilt(sos, audio)

        # ---- Envelope extraction params (identical) ----
        env_window = int(0.05 * fs)  # ~50 ms
        env_window = max(env_window, 1)
        env_hop = int(0.01 * fs)     # ~10 ms
        env_hop = max(env_hop, 1)
        hann = np.hanning(env_window).astype(np.float64)

        # ---- Sliding STI window (identical) ----
        frame_length = int(window_dur * fs)
        frame_step   = int(hop_dur * fs)
        num_frames = 1 + max(0, (len(audio) - frame_length) // frame_step)

        sti_values = []
        time_stamps = []

        # Also keep per-band MTI history for your band_sti output
        band_sti_values = {center: [] for center in valid_bands}

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_length
            if end > len(audio):
                break

            segment_sti = 0.0

            for center, Wk in zip(valid_bands, valid_weights):
                x_band = band_signals[center][start:end]
                power = x_band ** 2

                # Envelope via 'valid' conv then decimate every env_hop samples
                if len(power) < len(hann):
                    pad = len(hann) - len(power)
                    power_padded = np.pad(power, (0, pad), mode='constant', constant_values=0.0)
                else:
                    power_padded = power

                envelope = np.convolve(power_padded, hann, mode='valid')[::env_hop]
                envelope = np.clip(envelope, 0.0, None)

                E = envelope
                if E.size == 0:
                    continue
                sumE = float(np.sum(E))
                if sumE <= 1e-8:
                    continue

                # Modulation index at the 14 freqs (identical math)
                N = len(E)
                env_dt = env_hop / fs
                t = np.arange(N, dtype=np.float64) * env_dt

                M_f = []
                for f in self.mod_freqs:
                    phi = 2.0 * np.pi * f * t
                    comp = np.dot(E, np.exp(-1j * phi))
                    m_val = (2.0 * abs(comp)) / sumE
                    M_f.append(min(m_val, 1.0))
                M_f = np.array(M_f, dtype=np.float64)

                # SNR → TI (with identical clipping)
                eps = 1e-12
                M_sq = np.clip(M_f ** 2, 0.0, 1.0 - 1e-9)
                snr_values = 10.0 * np.log10((M_sq + eps) / (1.0 - M_sq + eps))
                snr_values = np.clip(snr_values, -15.0, 15.0)

                TI = (snr_values + 15.0) / 30.0
                MTI_k = float(np.mean(TI))

                segment_sti += Wk * MTI_k
                band_sti_values[center].append(MTI_k)

            # Center time of the window
            time_center_sec = (start + frame_length / 2.0) / fs
            time_stamps.append(time_center_sec)
            sti_values.append(segment_sti)

        # Convert to arrays (to match the script’s outputs)
        time_stamps = np.array(time_stamps, dtype=np.float64)
        sti_values = np.array(sti_values, dtype=np.float64)

        # Stats
        overall_sti = float(np.mean(sti_values)) if sti_values.size else 0.0
        sti_min = float(np.min(sti_values)) if sti_values.size else 0.0
        sti_max = float(np.max(sti_values)) if sti_values.size else 0.0
        sti_std = float(np.std(sti_values)) if sti_values.size else 0.0

        # Per-band mean MTI (informative; script doesn’t expose this, but keeping your API)
        band_sti_mean = {}
        for center in valid_bands:
            vals = band_sti_values.get(center, [])
            band_sti_mean[center] = float(np.mean(vals)) if len(vals) else 0.0

        # Temporal list (your format)
        temporal_sti = [{'time': float(t), 'sti': float(v)} for t, v in zip(time_stamps, sti_values)]

        return {
            'overall_sti': overall_sti,
            'sti_min': sti_min,
            'sti_max': sti_max,
            'sti_std': sti_std,
            'temporal_sti': temporal_sti,
            'time_stamps': time_stamps,
            'sti_values': sti_values,
            'band_sti': band_sti_mean,
            'band_centers': valid_bands,
            'band_weights': valid_weights.tolist(),
        }
