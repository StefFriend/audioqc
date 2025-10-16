"""
LUFS Analyzer Module for AudioQC
Handles LUFS (Loudness Units relative to Full Scale) and LNR calculations
"""

import numpy as np
import scipy.signal

class LUFSAnalyzer:
    """LUFS and LNR (LUFS to Noise Ratio) analyzer"""
    
    def __init__(self, audio, sr, hop_length):
        self.audio = audio
        self.sr = sr
        self.hop_length = hop_length
    
    def calculate_lufs_and_lnr(self, silence_mask=None):
        """Calculate LUFS and LNR (LUFS to Noise Ratio)"""
        # K-weighting filter coefficients
        # Stage 1: High shelf filter
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196
        K = np.tan(np.pi * f0 / self.sr)
        Vh = np.power(10.0, G / 20.0)
        Vb = np.power(Vh, 0.4996667741545416)
        
        a0 = 1.0 + K / Q + K * K
        b0 = (Vh + Vb * K / Q + K * K) / a0
        b1 = 2.0 * (K * K - Vh) / a0
        b2 = (Vh - Vb * K / Q + K * K) / a0
        a1 = 2.0 * (K * K - 1.0) / a0
        a2 = (1.0 - K / Q + K * K) / a0
        
        # Stage 2: High-pass filter
        f0 = 38.13547087602444
        Q = 0.5003270373238773
        K = np.tan(np.pi * f0 / self.sr)
        
        a0 = 1.0 + K / Q + K * K
        b0_hp = 1.0 / a0
        b1_hp = -2.0 / a0
        b2_hp = 1.0 / a0
        a1_hp = 2.0 * (K * K - 1.0) / a0
        a2_hp = (1.0 - K / Q + K * K) / a0
        
        # Apply K-weighting
        weighted = scipy.signal.lfilter([b0, b1, b2], [1.0, a1, a2], self.audio)
        weighted = scipy.signal.lfilter([b0_hp, b1_hp, b2_hp], [1.0, a1_hp, a2_hp], weighted)
        
        # Momentary loudness (400ms blocks, 100ms hop)
        block_size = int(0.4 * self.sr)
        hop_size = int(0.1 * self.sr)
        
        momentary_powers = []
        momentary_times = []
        
        for i in range(0, len(weighted) - block_size + 1, hop_size):
            block = weighted[i:i + block_size]
            power = np.mean(block ** 2)
            momentary_powers.append(power)
            momentary_times.append((i + block_size / 2) / self.sr)
        
        momentary_powers = np.array(momentary_powers)
        momentary_times = np.array(momentary_times)
        
        # Convert power to LUFS
        eps = 1e-10
        momentary_lufs = -0.691 + 10 * np.log10(momentary_powers + eps)
        
        # Short-term loudness (3s blocks, 1s hop)
        st_block_size = int(3.0 * self.sr)
        st_hop_size = int(1.0 * self.sr)
        
        short_term_powers = []
        short_term_times = []
        
        for i in range(0, max(1, len(weighted) - st_block_size + 1), st_hop_size):
            block = weighted[i:i + st_block_size]
            power = np.mean(block ** 2)
            short_term_powers.append(power)
            short_term_times.append((i + st_block_size / 2) / self.sr)
        
        short_term_powers = np.array(short_term_powers)
        short_term_times = np.array(short_term_times)
        short_term_lufs = -0.691 + 10 * np.log10(short_term_powers + eps)
        
        # Integrated loudness with gating
        absolute_gate_power = 10 ** ((-70 + 0.691) / 10)
        gated_powers = momentary_powers[momentary_powers >= absolute_gate_power]
        
        if len(gated_powers) > 0:
            ungated_mean_power = np.mean(gated_powers)
            relative_gate_power = ungated_mean_power / 10
            
            final_gate_power = max(absolute_gate_power, relative_gate_power)
            final_gated_powers = momentary_powers[momentary_powers >= final_gate_power]
            
            if len(final_gated_powers) > 0:
                integrated_power = np.mean(final_gated_powers)
            else:
                integrated_power = ungated_mean_power
        else:
            integrated_power = eps
        
        integrated_lufs = -0.691 + 10 * np.log10(integrated_power)
        
        # Calculate noise floor in LUFS
        if silence_mask is not None and np.any(silence_mask):
            # Expand silence mask to audio samples
            silence_samples = np.zeros(len(self.audio), dtype=bool)
            for i, is_silent in enumerate(silence_mask):
                start = i * self.hop_length
                end = min((i + 1) * self.hop_length, len(self.audio))
                silence_samples[start:end] = is_silent
            
            noise_weighted = weighted[silence_samples]
            noise_power = np.mean(noise_weighted ** 2) if len(noise_weighted) > 0 else eps
        else:
            # Use bottom 10% of weighted signal
            sorted_weighted = np.sort(np.abs(weighted))
            noise_weighted = sorted_weighted[:len(sorted_weighted)//10]
            noise_power = np.mean(noise_weighted ** 2) if len(noise_weighted) > 0 else eps
        
        noise_floor_lufs = -0.691 + 10 * np.log10(noise_power + eps)
        
        # Calculate LNR (LUFS to Noise Ratio)
        lnr = integrated_lufs - noise_floor_lufs
        
        # Temporal LNR
        temporal_lnr = []
        window_duration = 10  # seconds
        window_samples = int(window_duration * self.sr)
        n_windows = max(1, len(weighted) // window_samples)
        
        for i in range(n_windows):
            start = i * window_samples
            end = min((i + 1) * window_samples, len(weighted))
            window = weighted[start:end]
            
            if len(window) > 0:
                # Signal LUFS
                signal_power = np.mean(window ** 2)
                window_lufs = -0.691 + 10 * np.log10(signal_power + eps)
                
                # Noise LUFS (bottom 10% of window)
                sorted_window = np.sort(np.abs(window))
                noise_window = sorted_window[:len(sorted_window)//10]
                noise_power_window = np.mean(noise_window ** 2) if len(noise_window) > 0 else eps
                noise_lufs_window = -0.691 + 10 * np.log10(noise_power_window + eps)
                
                window_lnr = window_lufs - noise_lufs_window
                
                temporal_lnr.append({
                    'start': i * window_duration,
                    'end': min((i + 1) * window_duration, len(self.audio) / self.sr),
                    'lnr': float(window_lnr)
                })
        
        # Loudness range (LRA)
        if len(short_term_lufs) > 1:
            lra = np.percentile(short_term_lufs, 95) - np.percentile(short_term_lufs, 10)
        else:
            lra = 0.0
        
        # True peak: compute using small-window oversampling to avoid large allocations
        # Strategy: find candidate peaks, oversample small neighborhoods (e.g., 2048 samples) at 4x
        # This captures inter-sample peaks with minimal memory usage.
        up_factor = 4
        window = 2048  # samples around each candidate
        half_w = window // 2

        x = self.audio
        if x.size == 0:
            true_peak = 0.0
        else:
            # Candidate indices: top-N sample magnitudes
            N_candidates = int(max(8, min(64, x.size // max(1, (self.sr // 2)))))
            # Ensure at least some candidates
            N_candidates = max(8, N_candidates)

            # Use argpartition to get top candidates efficiently
            idx = np.argpartition(np.abs(x), -N_candidates)[-N_candidates:]
            idx.sort()

            tp = 0.0
            for i0 in idx:
                start = max(0, i0 - half_w)
                end = min(x.size, i0 + half_w)
                seg = x[start:end]
                if seg.size < 4:
                    tp = max(tp, float(np.max(np.abs(seg))))
                    continue
                # Oversample small segment by 4x using FFT-based resample
                try:
                    y = scipy.signal.resample(seg, seg.size * up_factor)
                    tp = max(tp, float(np.max(np.abs(y))))
                except Exception:
                    # Fallback: use original segment peak
                    tp = max(tp, float(np.max(np.abs(seg))))
            true_peak = tp

        true_peak_db = 20 * np.log10(true_peak + eps)
        
        return {
            'integrated': float(integrated_lufs),
            'momentary': momentary_lufs,
            'momentary_times': momentary_times,
            'short_term': short_term_lufs,
            'short_term_times': short_term_times,
            'max_momentary': float(np.max(momentary_lufs)) if len(momentary_lufs) > 0 else -70.0,
            'max_short_term': float(np.max(short_term_lufs)) if len(short_term_lufs) > 0 else -70.0,
            'lra': float(lra),
            'true_peak': float(true_peak),
            'true_peak_db': float(true_peak_db),
            'noise_floor_lufs': float(noise_floor_lufs),
            'lnr': float(lnr),
            'temporal_lnr': temporal_lnr
        }
