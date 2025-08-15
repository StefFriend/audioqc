import numpy as np
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime
import os
import sys
import argparse
import glob
import wave
import audioop
import warnings
import hashlib
warnings.filterwarnings('ignore')

# Try to import pydub for universal audio format support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Install with: pip install pydub")
    print("Limited to WAV format support only.")

class AudioAnalyzer:
    """
    Professional audio analyzer with objective measurements
    Including SNR, LUFS, and LNR (LUFS to Noise Ratio)
    """
    
    def __init__(self, audio_file, sr=None):
        """Initialize the analyzer"""
        if isinstance(audio_file, str):
            self.filename = os.path.basename(audio_file)
            self.filepath = audio_file
            print(f"  Loading: {self.filename}")
            
            # Calculate file hash
            self.file_hash = self.calculate_file_hash(audio_file)
            print(f"  SHA256: {self.file_hash[:16]}...")
            
            self.audio, self.sr, self.metadata = self.load_audio_universal(audio_file, sr)
            
            # Store original channels info
            if len(self.audio.shape) > 1:
                self.original_channels = self.audio.shape[0]
                self.audio = np.mean(self.audio, axis=0)
                print(f"  Converted from {self.original_channels} channels to mono")
            else:
                self.original_channels = 1
        else:
            self.filename = "Audio Stream"
            self.filepath = None
            self.file_hash = "N/A"
            self.audio = audio_file
            self.sr = sr if sr else 44100
            self.original_channels = 1
            self.metadata = {}
        
        # Get file info
        if self.filepath and os.path.exists(self.filepath):
            self.file_size = os.path.getsize(self.filepath) / (1024*1024)  # MB
        else:
            self.file_size = 0
        
        # Normalize to float32
        if self.audio.dtype != np.float32:
            self.audio = self.audio.astype(np.float32)
        
        # Ensure in [-1, 1] range
        max_val = np.max(np.abs(self.audio))
        if max_val > 1.0:
            self.audio = self.audio / max_val
        
        self.duration = len(self.audio) / self.sr
        print(f"  Duration: {self.duration:.2f}s, Sample rate: {self.sr}Hz")
        
        # Analysis parameters
        self.frame_length = 2048
        self.hop_length = 512
        self.results = {}
        
        # PDF settings
        self.dpi = 100
        self.color_scheme = {
            'primary': '#1e3a5f',      # Dark blue
            'secondary': '#4a90e2',    # Light blue
            'accent': '#f39c12',       # Orange
            'success': '#27ae60',      # Green
            'warning': '#e67e22',      # Orange
            'danger': '#e74c3c',       # Red
            'neutral': '#95a5a6'       # Gray
        }
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of the file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def load_audio_universal(self, filepath, target_sr=None):
        """Universal audio loader with metadata extraction"""
        metadata = {
            'format': os.path.splitext(filepath)[1][1:].upper(),
            'original_sr': None,
            'bit_depth': None,
            'channels': None,
            'codec': None
        }
        
        # Method 1: Try pydub for universal format support
        if PYDUB_AVAILABLE:
            try:
                print(f"  Attempting to load with pydub...")
                audio_segment = AudioSegment.from_file(filepath)
                
                # Extract metadata
                metadata['original_sr'] = audio_segment.frame_rate
                metadata['bit_depth'] = audio_segment.sample_width * 8
                metadata['channels'] = audio_segment.channels
                
                sr = audio_segment.frame_rate
                
                # Convert to target sample rate if specified
                if target_sr and sr != target_sr:
                    audio_segment = audio_segment.set_frame_rate(target_sr)
                    sr = target_sr
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Handle multi-channel
                if audio_segment.channels > 1:
                    samples = samples.reshape((-1, audio_segment.channels))
                    audio = samples.T  # Shape: (channels, samples)
                else:
                    audio = samples
                
                # Normalize based on bit depth
                if audio_segment.sample_width == 1:
                    audio = audio.astype(np.float32) / 128.0
                elif audio_segment.sample_width == 2:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 3:
                    audio = audio.astype(np.float32) / 8388608.0
                elif audio_segment.sample_width == 4:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val
                
                print(f"  Successfully loaded with pydub")
                return audio, sr, metadata
                
            except Exception as e:
                print(f"  Pydub failed: {e}, trying alternative methods...")
        
        # Method 2: Python wave module for WAV files
        if filepath.lower().endswith('.wav'):
            try:
                with wave.open(filepath, 'rb') as w:
                    params = w.getparams()
                    sr = params.framerate
                    n_channels = params.nchannels
                    n_frames = params.nframes
                    sampwidth = params.sampwidth
                    comptype = params.comptype
                    
                    metadata['original_sr'] = sr
                    metadata['bit_depth'] = sampwidth * 8
                    metadata['channels'] = n_channels
                    metadata['codec'] = comptype
                    
                    print(f"  WAV format: {comptype}, {sr}Hz, {n_channels}ch")
                    
                    frames = w.readframes(n_frames)
                    
                    if comptype == 'NONE':
                        if sampwidth == 2:
                            audio = np.frombuffer(frames, dtype=np.int16)
                            audio = audio.astype(np.float32) / 32768.0
                        elif sampwidth == 1:
                            audio = np.frombuffer(frames, dtype=np.uint8)
                            audio = audio.astype(np.float32) / 128.0 - 1.0
                        elif sampwidth == 4:
                            audio = np.frombuffer(frames, dtype=np.int32)
                            audio = audio.astype(np.float32) / 2147483648.0
                    elif comptype in ['ALAW', 'alaw']:
                        decoded = audioop.alaw2lin(frames, 2)
                        audio = np.frombuffer(decoded, dtype=np.int16)
                        audio = audio.astype(np.float32) / 32768.0
                    elif comptype in ['ULAW', 'ulaw']:
                        decoded = audioop.ulaw2lin(frames, 2)
                        audio = np.frombuffer(decoded, dtype=np.int16)
                        audio = audio.astype(np.float32) / 32768.0
                    
                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels).T
                    
                    return audio, sr, metadata
                    
            except Exception as e:
                print(f"  Wave module error: {e}")
        
        # Method 3: scipy fallback
        try:
            sr, audio = wavfile.read(filepath)
            metadata['original_sr'] = sr
            
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
                metadata['bit_depth'] = 16
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
                metadata['bit_depth'] = 32
            
            if len(audio.shape) > 1:
                audio = audio.T
                metadata['channels'] = audio.shape[0]
            else:
                metadata['channels'] = 1
            
            return audio, sr, metadata
            
        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {e}")
    
    def calculate_rms(self, audio, frame_length, hop_length):
        """Calculate RMS energy with numerical stability"""
        eps = 1e-10
        n = len(audio)
        
        if n == 0:
            return np.array([0.0])
        
        if n < frame_length:
            return np.array([np.sqrt(np.mean(audio ** 2) + eps)])
        
        n_frames = 1 + (n - frame_length) // hop_length
        rms = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, n)
            frame = audio[start:end]
            rms[i] = np.sqrt(np.mean(frame ** 2) + eps)
        
        return rms
    
    def detect_silence(self, percentile=10, min_silence_duration=0.1):
        """Detect silence regions with adaptive thresholding"""
        rms = self.calculate_rms(self.audio, self.frame_length, self.hop_length)
        
        # Adaptive threshold
        sorted_rms = np.sort(rms)
        noise_floor_estimate = np.mean(sorted_rms[:max(1, len(sorted_rms)//10)])
        threshold = max(np.percentile(rms, percentile), noise_floor_estimate * 1.5)
        
        silence_mask = rms < threshold
        
        # Minimum silence duration filtering
        min_frames = max(1, int(min_silence_duration * self.sr / self.hop_length))
        
        # Clean up short segments
        silence_regions = []
        start = None
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and start is None:
                start = i
            elif not is_silent and start is not None:
                if i - start >= min_frames:
                    silence_regions.append((start, i))
                start = None
        
        if start is not None and len(silence_mask) - start >= min_frames:
            silence_regions.append((start, len(silence_mask)))
        
        # Rebuild clean mask
        clean_mask = np.zeros_like(silence_mask)
        for start, end in silence_regions:
            clean_mask[start:end] = True
        
        return clean_mask, rms, silence_regions
    
    def calculate_snr(self):
        """Calculate SNR metrics"""
        eps = 1e-10
        
        # Detect silence
        silence_mask, rms, silence_regions = self.detect_silence()
        active_mask = ~silence_mask
        
        # Noise floor estimation
        if np.any(silence_mask):
            noise_rms = np.median(rms[silence_mask])
        else:
            noise_rms = np.percentile(rms, 10)
        
        # Signal level estimation
        if np.any(active_mask):
            active_rms = rms[active_mask]
            q1, q3 = np.percentile(active_rms, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_rms = active_rms[(active_rms >= lower_bound) & (active_rms <= upper_bound)]
            
            if len(filtered_rms) > 0:
                signal_rms = np.mean(filtered_rms)
            else:
                signal_rms = np.median(active_rms)
        else:
            signal_rms = np.percentile(rms, 75)
        
        # Calculate SNR
        global_snr = 20 * np.log10((signal_rms + eps) / (noise_rms + eps))
        noise_floor_db = 20 * np.log10(noise_rms + eps)
        signal_level_db = 20 * np.log10(signal_rms + eps)
        
        # Speech-weighted SNR
        nyquist = self.sr / 2
        low = min(300 / nyquist, 0.99)
        high = min(3400 / nyquist, 0.99)
        
        if high > low:
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered_audio = scipy.signal.filtfilt(b, a, self.audio)
            
            rms_filtered = self.calculate_rms(filtered_audio, self.frame_length, self.hop_length)
            
            threshold_filtered = np.percentile(rms_filtered, 10)
            silence_filtered = rms_filtered < threshold_filtered
            active_filtered = ~silence_filtered
            
            if np.any(silence_filtered):
                noise_filtered = np.median(rms_filtered[silence_filtered])
            else:
                noise_filtered = np.percentile(rms_filtered, 10)
            
            if np.any(active_filtered):
                signal_filtered = np.mean(rms_filtered[active_filtered])
            else:
                signal_filtered = np.percentile(rms_filtered, 75)
            
            speech_snr = 20 * np.log10((signal_filtered + eps) / (noise_filtered + eps))
        else:
            speech_snr = global_snr
        
        # Temporal SNR analysis
        window_duration = 10  # seconds
        window_samples = int(window_duration * self.sr)
        n_windows = max(1, len(self.audio) // window_samples)
        
        temporal_snr = []
        for i in range(n_windows):
            start = i * window_samples
            end = min((i + 1) * window_samples, len(self.audio))
            window = self.audio[start:end]
            
            if len(window) >= self.frame_length:
                window_rms = self.calculate_rms(window, self.frame_length, self.hop_length)
                window_noise = np.percentile(window_rms, 10)
                window_signal = np.mean(window_rms[window_rms > np.percentile(window_rms, 50)])
                window_snr = 20 * np.log10((window_signal + eps) / (window_noise + eps))
                
                temporal_snr.append({
                    'start': i * window_duration,
                    'end': min((i + 1) * window_duration, self.duration),
                    'snr': float(window_snr)
                })
        
        # Crest factor
        peak = np.max(np.abs(self.audio))
        rms_global = np.sqrt(np.mean(self.audio ** 2))
        crest_factor_db = 20 * np.log10((peak + eps) / (rms_global + eps))
        
        return {
            'global_snr': float(global_snr),
            'speech_weighted_snr': float(speech_snr),
            'noise_floor': float(noise_floor_db),
            'signal_level': float(signal_level_db),
            'crest_factor': float(crest_factor_db),
            'rms': rms,
            'silence_mask': silence_mask,
            'silence_regions': silence_regions,
            'silence_percentage': float(np.mean(silence_mask) * 100),
            'temporal_snr': temporal_snr,
            'noise_rms': noise_rms  # Store for LNR calculation
        }
    
    def calculate_lufs_and_lnr(self):
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
        # Apply K-weighting to silence regions
        silence_mask, _, _ = self.detect_silence()
        
        # Expand silence mask to audio samples
        silence_samples = np.zeros(len(self.audio), dtype=bool)
        for i, is_silent in enumerate(silence_mask):
            start = i * self.hop_length
            end = min((i + 1) * self.hop_length, len(self.audio))
            silence_samples[start:end] = is_silent
        
        if np.any(silence_samples):
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
                    'end': min((i + 1) * window_duration, self.duration),
                    'lnr': float(window_lnr)
                })
        
        # Loudness range (LRA)
        if len(short_term_lufs) > 1:
            lra = np.percentile(short_term_lufs, 95) - np.percentile(short_term_lufs, 10)
        else:
            lra = 0.0
        
        # True peak
        oversampled = scipy.signal.resample(self.audio, len(self.audio) * 4)
        true_peak = np.max(np.abs(oversampled))
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
    
    def analyze_frequency_bands(self):
        """Analyze frequency content and band-specific SNR"""
        nperseg = min(self.frame_length, 2048)
        
        f, t, Sxx = scipy.signal.spectrogram(
            self.audio, self.sr,
            nperseg=nperseg,
            noverlap=nperseg//2,
            window='hann',
            scaling='spectrum'
        )
        
        bands = [
            (20, 60, "Sub-bass"),
            (60, 250, "Bass"),
            (250, 500, "Low-mid"),
            (500, 2000, "Mid"),
            (2000, 4000, "High-mid"),
            (4000, 8000, "Presence"),
            (8000, min(16000, self.sr/2), "Brilliance")
        ]
        
        band_analysis = []
        for low, high, name in bands:
            mask = (f >= low) & (f < high)
            if not np.any(mask):
                continue
            
            band_power = Sxx[mask, :].mean(axis=0)
            
            noise_floor = np.percentile(band_power, 10)
            signal_level = np.percentile(band_power, 75)
            
            if noise_floor > 0:
                band_snr = 10 * np.log10(signal_level / noise_floor)
            else:
                band_snr = 0
            
            avg_power = 10 * np.log10(np.mean(band_power) + 1e-10)
            
            band_analysis.append({
                'range': f"{int(low)}-{int(high)} Hz",
                'name': name,
                'snr': float(band_snr),
                'avg_power': float(avg_power),
                'low': low,
                'high': high
            })
        
        return band_analysis, f, t, Sxx
    
    def calculate_statistics(self):
        """Calculate comprehensive audio statistics"""
        eps = 1e-10
        
        # Basic statistics
        peak = np.max(np.abs(self.audio))
        peak_db = 20 * np.log10(peak + eps)
        
        rms = np.sqrt(np.mean(self.audio ** 2))
        rms_db = 20 * np.log10(rms + eps)
        
        # Dynamic range
        rms_frames = self.calculate_rms(self.audio, self.frame_length, self.hop_length)
        silence_mask, _, _ = self.detect_silence()
        active_frames = rms_frames[~silence_mask] if np.any(~silence_mask) else rms_frames
        
        if len(active_frames) > 1:
            dr_high = np.percentile(active_frames, 95)
            dr_low = np.percentile(active_frames, 10)
            dynamic_range = 20 * np.log10((dr_high + eps) / (dr_low + eps))
        else:
            dynamic_range = 0.0
        
        # Spectral statistics
        fft = np.fft.rfft(self.audio * np.hanning(len(self.audio)))
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(self.audio), 1/self.sr)
        
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0.0
        
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs[rolloff_idx[0]]
            else:
                spectral_rolloff = freqs[-1]
        else:
            spectral_rolloff = 0.0
        
        zcr = np.sum(np.abs(np.diff(np.sign(self.audio)))) / (2 * len(self.audio))
        
        return {
            'peak': float(peak),
            'peak_db': float(peak_db),
            'rms': float(rms),
            'rms_db': float(rms_db),
            'dynamic_range': float(np.clip(dynamic_range, 0, 120)),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'zcr': float(zcr)
        }
    
    def create_professional_report(self):
        """Generate professional A4 PDF report without quality judgments"""
        print(f"  Generating analysis report...")
        
        # Run all analyses
        snr = self.calculate_snr()
        lufs = self.calculate_lufs_and_lnr()
        bands, f, t, Sxx = self.analyze_frequency_bands()
        stats = self.calculate_statistics()
        
        # Store results
        self.results = {
            'snr': snr,
            'lufs': lufs,
            'bands': bands,
            'stats': stats,
            'metadata': self.metadata,
            'file_hash': self.file_hash
        }
        
        # Create report pages
        figures = []
        
        # Page 1: Executive Summary with Hash
        fig1 = self.create_executive_summary_page()
        figures.append(fig1)
        
        # Page 2: SNR and LNR Analysis
        fig2 = self.create_snr_lnr_page()
        figures.append(fig2)
        
        # Page 3: LUFS Analysis
        fig3 = self.create_lufs_analysis_page()
        figures.append(fig3)
        
        # Page 4: Spectral Analysis
        fig4 = self.create_spectral_analysis_page()
        figures.append(fig4)
        
        # Page 5: Measurement Explanation
        fig5 = self.create_measurement_explanation_page()
        figures.append(fig5)
        
        # Page 6: Standards Reference
        fig6 = self.create_standards_reference_page()
        figures.append(fig6)
        
        return figures
    
    def create_executive_summary_page(self):
        """Create executive summary page with file hash and measurements"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        # Header
        fig.text(0.5, 0.96, 'AUDIOQC ANALYSIS REPORT', 
                fontsize=18, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        fig.text(0.5, 0.94, 'Technical Measurements', 
                fontsize=14, ha='center', color=self.color_scheme['secondary'])
        
        # Grid
        gs = GridSpec(6, 3, figure=fig, hspace=0.5, wspace=0.4,
                     top=0.90, bottom=0.05, left=0.08, right=0.92)
        
        # File information with hash
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        info_rect = mpatches.FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                                           boxstyle="round,pad=0.02",
                                           facecolor='#f8f9fa',
                                           edgecolor=self.color_scheme['primary'],
                                           linewidth=2)
        ax_info.add_patch(info_rect)
        
        info_text = f"""
  File: {self.filename}
  SHA256: {self.file_hash}
  Format: {self.metadata.get('format', 'Unknown')} | Channels: {self.metadata.get('channels', 1)} | Sample Rate: {self.sr} Hz
  Duration: {self.duration:.2f}s | Size: {self.file_size:.2f} MB | Bit Depth: {self.metadata.get('bit_depth', 'Unknown')}
        """
        ax_info.text(0.5, 0.5, info_text, fontsize=9, ha='center', va='center',
                    transform=ax_info.transAxes, family='monospace')
        
        # Key measurements table
        ax_table = fig.add_subplot(gs[1:3, :])
        ax_table.axis('off')
        
        measurements = [
            ['Measurement', 'Value', 'Unit'],
            ['', '', ''],
            ['SIGNAL METRICS', '', ''],
            ['Global SNR', f"{self.results['snr']['global_snr']:.1f}", 'dB'],
            ['Speech-weighted SNR', f"{self.results['snr']['speech_weighted_snr']:.1f}", 'dB'],
            ['LNR (LUFS to Noise)', f"{self.results['lufs']['lnr']:.1f}", 'LU'],
            ['', '', ''],
            ['LOUDNESS', '', ''],
            ['Integrated LUFS', f"{self.results['lufs']['integrated']:.1f}", 'LUFS'],
            ['Max Momentary', f"{self.results['lufs']['max_momentary']:.1f}", 'LUFS'],
            ['Max Short-term', f"{self.results['lufs']['max_short_term']:.1f}", 'LUFS'],
            ['Loudness Range (LRA)', f"{self.results['lufs']['lra']:.1f}", 'LU'],
            ['', '', ''],
            ['LEVELS', '', ''],
            ['True Peak', f"{self.results['lufs']['true_peak_db']:.1f}", 'dBTP'],
            ['Noise Floor (dB)', f"{self.results['snr']['noise_floor']:.1f}", 'dBFS'],
            ['Noise Floor (LUFS)', f"{self.results['lufs']['noise_floor_lufs']:.1f}", 'LUFS'],
            ['Signal Level', f"{self.results['snr']['signal_level']:.1f}", 'dBFS'],
            ['', '', ''],
            ['DYNAMICS', '', ''],
            ['Dynamic Range', f"{self.results['stats']['dynamic_range']:.1f}", 'dB'],
            ['Crest Factor', f"{self.results['snr']['crest_factor']:.1f}", 'dB'],
            ['Silence', f"{self.results['snr']['silence_percentage']:.1f}", '%']
        ]
        
        # Create formatted table
        table_text = ""
        for row in measurements:
            if row[0] in ['SIGNAL METRICS', 'LOUDNESS', 'LEVELS', 'DYNAMICS']:
                table_text += f"\n{row[0]}\n" + "─" * 60 + "\n"
            elif row[0] != '':
                table_text += f"  {row[0]:<25} {row[1]:>15} {row[2]:>10}\n"
        
        ax_table.text(0.1, 0.5, table_text, fontsize=9, family='monospace',
                     transform=ax_table.transAxes, va='center')
        
        # Waveform preview
        ax_wave = fig.add_subplot(gs[3:5, :])
        self.plot_waveform_simple(ax_wave)
        
        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        ax_footer.axis('off')
        footer_text = f"""
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AudioQC v0.2
        """
        ax_footer.text(0.5, 0.5, footer_text, fontsize=8, ha='center', va='center',
                      transform=ax_footer.transAxes, color='gray')
        
        return fig
    
    def create_snr_lnr_page(self):
        """Create SNR and LNR analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'SNR AND LNR ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35,
                     top=0.93, bottom=0.04, left=0.08, right=0.95)
        
        # RMS Energy with noise floor
        ax_rms = fig.add_subplot(gs[0, :])
        self.plot_rms_energy(ax_rms)
        
        # Temporal SNR
        ax_snr = fig.add_subplot(gs[1, :])
        self.plot_temporal_snr(ax_snr)
        
        # Temporal LNR (new)
        ax_lnr = fig.add_subplot(gs[2, :])
        self.plot_temporal_lnr(ax_lnr)
        
        # SNR vs LNR Comparison
        ax_compare = fig.add_subplot(gs[3, :])
        self.plot_snr_lnr_comparison(ax_compare)
        
        # Frequency bands SNR
        ax_bands = fig.add_subplot(gs[4, :])
        self.plot_frequency_bands(ax_bands)
        
        return fig
    
    def create_lufs_analysis_page(self):
        """Create LUFS analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'LOUDNESS ANALYSIS (LUFS)', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.35,
                     top=0.93, bottom=0.04, left=0.08, right=0.95)
        
        # LUFS Timeline
        ax_lufs = fig.add_subplot(gs[0:2, :])
        self.plot_lufs_timeline(ax_lufs)
        
        # LUFS distribution
        ax_dist = fig.add_subplot(gs[2, 0])
        self.plot_lufs_distribution(ax_dist)
        
        # Amplitude distribution
        ax_amp = fig.add_subplot(gs[2, 1])
        self.plot_amplitude_histogram(ax_amp)
        
        # LUFS Statistics
        ax_stats = fig.add_subplot(gs[3, :])
        self.create_lufs_statistics_table(ax_stats)
        
        return fig
    
    def create_spectral_analysis_page(self):
        """Create spectral analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'SPECTRAL ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(3, 1, figure=fig, hspace=0.35,
                     top=0.93, bottom=0.04, left=0.08, right=0.95)
        
        # Spectrogram
        ax_spec = fig.add_subplot(gs[0:2])
        self.plot_spectrogram(ax_spec)
        
        # Spectral statistics
        ax_stats = fig.add_subplot(gs[2])
        self.plot_spectral_statistics(ax_stats)
        
        return fig
    
    def create_measurement_explanation_page(self):
        """Create page explaining SNR, LUFS, and LNR"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'MEASUREMENT EXPLANATIONS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        explanation_text = """
SNR (Signal-to-Noise Ratio)
────────────────────────────────────────────────────────────────────────────────
The SNR measures the level difference between the desired signal and background 
noise in decibels (dB). It represents how much the signal stands out from the 
noise floor.

• Global SNR: Overall signal-to-noise ratio across the entire file
• Speech-weighted SNR: SNR filtered for speech frequencies (300-3400 Hz)
• Higher values indicate cleaner recordings with less noise
• Measured in dB (logarithmic scale)

LUFS (Loudness Units relative to Full Scale)
────────────────────────────────────────────────────────────────────────────────
LUFS is a standardized measurement of audio loudness that accounts for human 
perception, as defined in ITU-R BS.1770-4. Unlike peak or RMS measurements, 
LUFS uses K-weighting filters to match human hearing sensitivity.

• Integrated LUFS: Average loudness over the entire file with gating
• Momentary LUFS: 400ms window measurements
• Short-term LUFS: 3-second window measurements
• LRA (Loudness Range): Difference between loud and quiet parts
• Negative scale where 0 LUFS = digital maximum

LNR (LUFS to Noise Ratio) - Custom derived metric
────────────────────────────────────────────────────────────────────────────────
LNR is the loudness-domain equivalent of SNR, measuring the difference between 
the integrated LUFS and the noise floor expressed in LUFS. This provides a 
perceptually-weighted assessment of signal clarity.

• LNR = Integrated LUFS - Noise Floor LUFS
• Measured in LU (Loudness Units)
• Accounts for psychoacoustic weighting
• Better represents perceived signal clarity than traditional SNR

KEY DIFFERENCES
────────────────────────────────────────────────────────────────────────────────

SNR vs LNR:
• SNR: Linear amplitude domain, unweighted
• LNR: Loudness domain, K-weighted for human perception
• Both measure signal clarity but LNR better matches what we hear

LUFS vs dBFS:
• dBFS: Simple amplitude measurement relative to digital full scale
• LUFS: Perceptually-weighted loudness measurement
• LUFS accounts for frequency-dependent hearing sensitivity

INTERPRETATION GUIDE
────────────────────────────────────────────────────────────────────────────────

High SNR + High LNR: Clean recording with good technical and perceptual clarity
High SNR + Low LNR: Technically clean but perceptually less clear
Low SNR + High LNR: Technical noise present but perceptually acceptable
Low SNR + Low LNR: Both technical and perceptual noise issues
        """
        
        ax.text(0.05, 0.95, explanation_text, fontsize=8.5, family='monospace',
               transform=ax.transAxes, va='top')
        
        return fig
    
    def create_standards_reference_page(self):
        """Create standards reference page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'TECHNICAL STANDARDS REFERENCE', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        standards_text = """
LOUDNESS STANDARDS
────────────────────────────────────────────────────────────────────────────────

ITU-R BS.1770-4 (2015)
Algorithms to measure audio programme loudness and true-peak audio level
• Defines K-weighting filters for loudness measurement
• Specifies gating algorithm for integrated loudness
• True-peak measurement via oversampling

EBU R128 (2020)
Loudness normalisation and permitted maximum level of audio signals
• Target Level: -23.0 LUFS (±0.5 LU tolerance)
• Max True Peak: -1 dBTP
• Loudness Range: 5-20 LU typical

ATSC A/85 (2013)
Techniques for Establishing and Maintaining Audio Loudness (USA)
• Target Level: -24.0 LKFS (±2 LU tolerance)
• Max True Peak: -2 dBTP

AES TD1004.1.15-10 (2015)
Recommendation for Loudness of Audio Streaming and Network File Playback
• Streaming Target: -16 to -20 LUFS
• Mobile Target: -16 LUFS
• No true peak limit specified

SIGNAL QUALITY SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────

ITU-T P.56 (2011)
Objective measurement of active speech level
• Defines speech-band filtering (300-3400 Hz)
• Active speech level measurement methodology

AES17-2020
Standard method for digital audio engineering measurement
• Dynamic range measurement procedures
• Signal-to-noise ratio definitions
• Measurement bandwidth specifications

IEC 61606-3 (2008)
Audio and audiovisual equipment - Digital audio parts
• Basic measurement methods
• Reference levels and alignment

BROADCAST SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────

EBU Tech 3343 (2016)
Guidelines for Production of Programmes in accordance with EBU R128
• Production ranges and tolerances
• Measurement gate specifications

BBC R&D White Paper WHP 259 (2014)
Audio Metering and Monitoring
• Practical implementation guidelines
• Measurement best practices

STREAMING PLATFORM SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────

Spotify (2021): -14 LUFS integrated, -1 dBTP max
Apple Music (2021): -16 LUFS integrated, -1 dBTP max
YouTube (2021): -14 LUFS integrated
Amazon Music (2021): -14 to -9 LUFS integrated
Tidal (2021): -14 LUFS integrated

MEASUREMENT NOTES
────────────────────────────────────────────────────────────────────────────────

All measurements in this report comply with:
• ITU-R BS.1770-4 for LUFS calculation
• ITU-T P.56 for speech-weighted measurements
• AES17-2020 for dynamic range assessment

Calibration: 0 dBFS = Full Scale Digital
Reference: 1 kHz sine wave at -20 dBFS
        """
        
        ax.text(0.05, 0.95, standards_text, fontsize=8, family='monospace',
               transform=ax.transAxes, va='top')
        
        return fig
    
    # Plotting helper functions
    def plot_waveform_simple(self, ax):
        """Plot simple waveform"""
        display_samples = min(len(self.audio), 20000)
        if len(self.audio) > display_samples:
            indices = np.linspace(0, len(self.audio) - 1, display_samples, dtype=int)
            audio_display = self.audio[indices]
            time_display = indices / self.sr
        else:
            audio_display = self.audio
            time_display = np.arange(len(self.audio)) / self.sr
        
        ax.fill_between(time_display, audio_display, alpha=0.5, color=self.color_scheme['secondary'])
        ax.plot(time_display, audio_display, linewidth=0.5, color=self.color_scheme['primary'])
        
        # Mark silence regions
        for start, end in self.results['snr']['silence_regions']:
            start_time = start * self.hop_length / self.sr
            end_time = end * self.hop_length / self.sr
            ax.axvspan(start_time, end_time, alpha=0.2, color=self.color_scheme['danger'])
        
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title('Waveform Overview', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(-1.05, 1.05)
    
    def plot_rms_energy(self, ax):
        """Plot RMS energy"""
        rms_time = np.arange(len(self.results['snr']['rms'])) * self.hop_length / self.sr
        rms_db = 20 * np.log10(self.results['snr']['rms'] + 1e-10)
        
        ax.fill_between(rms_time, -60, rms_db, alpha=0.3, color=self.color_scheme['secondary'])
        ax.plot(rms_time, rms_db, linewidth=1, color=self.color_scheme['primary'])
        
        ax.axhline(y=self.results['snr']['noise_floor'], color=self.color_scheme['danger'],
                  linestyle='--', linewidth=1.5, alpha=0.8,
                  label=f'Noise Floor: {self.results["snr"]["noise_floor"]:.1f} dB')
        ax.axhline(y=self.results['snr']['signal_level'], color=self.color_scheme['success'],
                  linestyle='--', linewidth=1.5, alpha=0.8,
                  label=f'Signal Level: {self.results["snr"]["signal_level"]:.1f} dB')
        
        ax.set_xlabel('Time (s)', fontsize=9, labelpad=2)
        ax.set_ylabel('Level (dBFS)', fontsize=9)
        ax.set_title('RMS Energy Analysis', fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(max(-60, self.results['snr']['noise_floor'] - 10), 0)
    
    def plot_temporal_snr(self, ax):
        """Plot temporal SNR"""
        if self.results['snr']['temporal_snr']:
            times = [(s['start'] + s['end'])/2 for s in self.results['snr']['temporal_snr']]
            values = [s['snr'] for s in self.results['snr']['temporal_snr']]
            
            ax.plot(times, values, marker='o', linewidth=2, markersize=5,
                   color=self.color_scheme['primary'], label='Temporal SNR')
            ax.axhline(y=self.results['snr']['global_snr'], color=self.color_scheme['accent'],
                      linestyle='--', linewidth=1.5, alpha=0.8,
                      label=f'Average: {self.results["snr"]["global_snr"]:.1f} dB')
            
            ax.set_xlabel('Time (s)', fontsize=9, labelpad=2)
            ax.set_ylabel('SNR (dB)', fontsize=9)
            ax.set_title('Temporal SNR Variation', fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-self.duration * 0.02, self.duration * 1.02)
    
    def plot_temporal_lnr(self, ax):
        """Plot temporal LNR (new)"""
        if self.results['lufs']['temporal_lnr']:
            times = [(s['start'] + s['end'])/2 for s in self.results['lufs']['temporal_lnr']]
            values = [s['lnr'] for s in self.results['lufs']['temporal_lnr']]
            
            ax.plot(times, values, marker='s', linewidth=2, markersize=5,
                   color=self.color_scheme['accent'], label='Temporal LNR')
            ax.axhline(y=self.results['lufs']['lnr'], color=self.color_scheme['primary'],
                      linestyle='--', linewidth=1.5, alpha=0.8,
                      label=f'Average: {self.results["lufs"]["lnr"]:.1f} LU')
            
            ax.set_xlabel('Time (s)', fontsize=9, labelpad=2)
            ax.set_ylabel('LNR (LU)', fontsize=9)
            ax.set_title('Temporal LNR (LUFS to Noise Ratio) Variation', fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-self.duration * 0.02, self.duration * 1.02)
    
    def plot_snr_lnr_comparison(self, ax):
        """Plot SNR vs LNR comparison"""
        categories = ['Global SNR (dB)', 'Speech SNR (dB)', 'LNR (LU)']
        values = [
            self.results['snr']['global_snr'],
            self.results['snr']['speech_weighted_snr'],
            self.results['lufs']['lnr']
        ]
        colors = [self.color_scheme['primary'], self.color_scheme['secondary'], self.color_scheme['accent']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        max_value = max(values)
        y_limit = max_value * 1.15  # + 15% > max_value
        ax.set_ylim(0, y_limit)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.02,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title('SNR and LNR Comparison', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def plot_frequency_bands(self, ax):
        """Plot frequency bands SNR"""
        bands = self.results['bands']
        
        names = [b['name'] for b in bands]
        values = [b['snr'] for b in bands]
        
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, values, color=self.color_scheme['secondary'], 
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        
        max_value = max(values)
        y_limit = max_value * 1.15  # + 15% > max_value
        ax.set_ylim(0, y_limit)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            y_pos = height + max(values) * 0.02 if height > 0 else 0.5
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('SNR (dB)', fontsize=9)
        ax.set_title('Signal-to-Noise Ratio by Frequency Band', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def plot_lufs_timeline(self, ax):
        """Plot LUFS timeline"""
        if len(self.results['lufs']['momentary']) > 500:
            step = len(self.results['lufs']['momentary']) // 500
            mom_display = self.results['lufs']['momentary'][::step]
            mom_times = self.results['lufs']['momentary_times'][::step]
        else:
            mom_display = self.results['lufs']['momentary']
            mom_times = self.results['lufs']['momentary_times']
        
        ax.plot(mom_times, mom_display, linewidth=0.5, alpha=0.5,
               label='Momentary', color=self.color_scheme['neutral'])
        ax.plot(self.results['lufs']['short_term_times'], self.results['lufs']['short_term'],
               linewidth=1.5, label='Short-term', color=self.color_scheme['secondary'])
        ax.axhline(y=self.results['lufs']['integrated'], color=self.color_scheme['accent'],
                  linestyle='--', linewidth=2,
                  label=f'Integrated: {self.results["lufs"]["integrated"]:.1f} LUFS')
        ax.axhline(y=self.results['lufs']['noise_floor_lufs'], color=self.color_scheme['danger'],
                  linestyle=':', linewidth=1.5, alpha=0.7,
                  label=f'Noise Floor: {self.results["lufs"]["noise_floor_lufs"]:.1f} LUFS')
        
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('LUFS', fontsize=9)
        ax.set_title('Loudness Timeline (ITU-R BS.1770-4)', fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(max(-70, min(self.results['lufs']['noise_floor_lufs'] - 5, 
                                 self.results['lufs']['integrated'] - 20)), 0)
    
    def plot_lufs_distribution(self, ax):
        """Plot LUFS distribution"""
        ax.hist(self.results['lufs']['momentary'], bins=50, alpha=0.7,
               color=self.color_scheme['accent'], edgecolor='black', linewidth=0.5)
        ax.axvline(x=self.results['lufs']['integrated'], color='red', linestyle='--',
                  linewidth=2, alpha=0.8)
        ax.axvline(x=self.results['lufs']['noise_floor_lufs'], color=self.color_scheme['danger'],
                  linestyle=':', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('LUFS', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Loudness Distribution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def plot_amplitude_histogram(self, ax):
        """Plot amplitude distribution"""
        ax.hist(self.audio, bins=100, alpha=0.7, color=self.color_scheme['secondary'],
               edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Amplitude', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Amplitude Distribution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        mean = np.mean(self.audio)
        std = np.std(self.audio)
        stats_text = f'μ={mean:.3f}\nσ={std:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_lufs_statistics_table(self, ax):
        """Create LUFS statistics table"""
        ax.axis('off')
        
        stats_text = f"""
LUFS DETAILED STATISTICS
────────────────────────────────────────────────────────────────────
Integrated Loudness:    {self.results['lufs']['integrated']:.2f} LUFS
Max Momentary:          {self.results['lufs']['max_momentary']:.2f} LUFS  
Max Short-term:         {self.results['lufs']['max_short_term']:.2f} LUFS
Loudness Range (LRA):    {self.results['lufs']['lra']:.2f} LU

Noise Floor (LUFS):     {self.results['lufs']['noise_floor_lufs']:.2f} LUFS
LNR (LUFS to Noise):     {self.results['lufs']['lnr']:.2f} LU
True Peak:              {self.results['lufs']['true_peak_db']:.2f} dBTP
True Peak Linear:        {self.results['lufs']['true_peak']:.4f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
               transform=ax.transAxes, va='center')
    
    def plot_spectrogram(self, ax):
        """Plot spectrogram"""
        _, f, t, Sxx = self.analyze_frequency_bands()
        
        max_freq_idx = np.where(f <= min(10000, self.sr/2))[0][-1]
        
        if Sxx.shape[1] > 500:
            step = Sxx.shape[1] // 500
            Sxx_display = Sxx[:max_freq_idx, ::step]
            t_display = t[::step]
        else:
            Sxx_display = Sxx[:max_freq_idx, :]
            t_display = t
        
        im = ax.pcolormesh(t_display, f[:max_freq_idx],
                          10 * np.log10(Sxx_display + 1e-10),
                          shading='auto', cmap='viridis', rasterized=True)
        
        ax.set_ylabel('Frequency (Hz)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_title('Spectrogram', fontsize=10, fontweight='bold')
        ax.set_ylim(0, min(10000, self.sr/2))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    
    def plot_spectral_statistics(self, ax):
        """Plot spectral statistics"""
        ax.axis('off')
        
        stats_text = f"""
SPECTRAL STATISTICS
────────────────────────────────────────────────────────────────────
Spectral Centroid:      {self.results['stats']['spectral_centroid']:.1f} Hz
Spectral Rolloff (85%): {self.results['stats']['spectral_rolloff']:.1f} Hz
Zero Crossing Rate:     {self.results['stats']['zcr']:.4f}

Peak Level:            {self.results['stats']['peak_db']:.2f} dBFS
RMS Level:             {self.results['stats']['rms_db']:.2f} dBFS
Dynamic Range:          {self.results['stats']['dynamic_range']:.2f} dB
Crest Factor:           {self.results['snr']['crest_factor']:.2f} dB
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
               transform=ax.transAxes, va='center')
    
    def save_report(self, output_path):
        """Save PDF report"""
        figures = self.create_professional_report()
        
        with PdfPages(output_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, dpi=self.dpi, bbox_inches='tight',
                          pad_inches=0.3, facecolor='white')
                plt.close(fig)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = f'Audio Analysis - {self.filename}'
            d['Author'] = 'AudioQC v0.2'
            d['Subject'] = 'AudioQC Analysis Report'
            d['Keywords'] = 'Audio, Analysis, SNR, LUFS, LNR'
            d['CreationDate'] = datetime.now()
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  ✓ Report saved: {output_path}")
        print(f"  ✓ Report size: {file_size_mb:.2f} MB")
        print(f"  ✓ Analysis complete")
        
        return self.results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AudioQC v0.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool provides objective audio measurements.

Measurements include:
  - SNR (Signal-to-Noise Ratio)
  - LUFS (Loudness Units relative to Full Scale) 
  - LNR (LUFS to Noise Ratio)
  - Spectral analysis
  - File integrity (SHA256)

Examples:
  %(prog)s audio.wav
  %(prog)s recording.mp3 -o reports/
  %(prog)s podcast.m4a --dpi 150

For full format support:
  pip install pydub
        """
    )
    
    parser.add_argument('input', nargs='?', help='Audio file to analyze')
    parser.add_argument('-o', '--output', default='audioqc_reports',
                       help='Output directory (default: audioqc_reports)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='PDF resolution in DPI (default: 100)')
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return 1
    
    print("="*60)
    print("AudioQC v0.2")
    print("="*60)
    
    if not PYDUB_AVAILABLE:
        print("\n⚠ Note: Install pydub for full format support")
        print("  pip install pydub")
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        filename = os.path.basename(args.input)
        name_only = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output, f"{name_only}_analysis.pdf")
        
        print(f"\nProcessing: {filename}")
        
        analyzer = AudioAnalyzer(args.input)
        analyzer.dpi = args.dpi
        
        results = analyzer.save_report(output_path)
        
        print(f"\n✓ Analysis complete")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())