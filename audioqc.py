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
warnings.filterwarnings('ignore')

# Try to import pydub for universal audio format support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Install with: pip install pydub")
    print("Limited to WAV format support only.")

class ProfessionalAudioAnalyzer:
    """
    Professional audio analyzer with verified calculations
    and enhanced A4 PDF reporting
    """
    
    def __init__(self, audio_file, sr=None):
        """Initialize the analyzer"""
        if isinstance(audio_file, str):
            self.filename = os.path.basename(audio_file)
            self.filepath = audio_file
            print(f"  Loading: {self.filename}")
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
        
        # Analysis parameters (verified for accuracy)
        self.frame_length = 2048
        self.hop_length = 512
        self.results = {}
        
        # PDF settings for professional output
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
            # For very short audio, use the whole signal
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
        
        # Adaptive threshold based on noise floor estimation
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
        """Calculate verified SNR metrics with robust estimators"""
        eps = 1e-10
        
        # Detect silence
        silence_mask, rms, silence_regions = self.detect_silence()
        active_mask = ~silence_mask
        
        # Noise floor estimation (use median for robustness)
        if np.any(silence_mask):
            noise_rms = np.median(rms[silence_mask])
        else:
            noise_rms = np.percentile(rms, 10)
        
        # Signal level estimation (trimmed mean for robustness)
        if np.any(active_mask):
            active_rms = rms[active_mask]
            # Remove outliers using IQR method
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
        
        # Speech-weighted SNR (ITU-T P.56 weighting)
        nyquist = self.sr / 2
        low = min(300 / nyquist, 0.99)
        high = min(3400 / nyquist, 0.99)
        
        if high > low:  # Only if valid frequency range
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered_audio = scipy.signal.filtfilt(b, a, self.audio)
            
            # Recalculate RMS for filtered signal
            rms_filtered = self.calculate_rms(filtered_audio, self.frame_length, self.hop_length)
            
            # Recalculate silence mask for filtered signal
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
        
        # Calculate crest factor
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
            'temporal_snr': temporal_snr
        }
    
    def calculate_lufs(self):
        """Calculate LUFS following ITU-R BS.1770-4 standard"""
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
        block_size = int(0.4 * self.sr)  # 400ms
        hop_size = int(0.1 * self.sr)    # 100ms
        
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
        # Absolute gate at -70 LUFS
        absolute_gate_power = 10 ** ((-70 + 0.691) / 10)
        gated_powers = momentary_powers[momentary_powers >= absolute_gate_power]
        
        if len(gated_powers) > 0:
            # Relative gate at -10 LU below ungated mean
            ungated_mean_power = np.mean(gated_powers)
            relative_gate_power = ungated_mean_power / 10  # -10 LU = factor of 10
            
            # Final gating
            final_gate_power = max(absolute_gate_power, relative_gate_power)
            final_gated_powers = momentary_powers[momentary_powers >= final_gate_power]
            
            if len(final_gated_powers) > 0:
                integrated_power = np.mean(final_gated_powers)
            else:
                integrated_power = ungated_mean_power
        else:
            integrated_power = eps
        
        integrated_lufs = -0.691 + 10 * np.log10(integrated_power)
        
        # Loudness range (LRA)
        if len(short_term_lufs) > 1:
            lra = np.percentile(short_term_lufs, 95) - np.percentile(short_term_lufs, 10)
        else:
            lra = 0.0
        
        # True peak (oversampled peak detection)
        # Simple 4x oversampling for true peak estimation
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
            'true_peak_db': float(true_peak_db)
        }
    
    def analyze_frequency_bands(self):
        """Analyze frequency content and band-specific SNR"""
        # Use optimized FFT size
        nperseg = min(self.frame_length, 2048)
        
        f, t, Sxx = scipy.signal.spectrogram(
            self.audio, self.sr,
            nperseg=nperseg,
            noverlap=nperseg//2,
            window='hann',
            scaling='spectrum'
        )
        
        # Professional frequency bands
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
            
            # Robust SNR estimation for band
            noise_floor = np.percentile(band_power, 10)
            signal_level = np.percentile(band_power, 75)
            
            if noise_floor > 0:
                band_snr = 10 * np.log10(signal_level / noise_floor)
            else:
                band_snr = 0
            
            # Average power in band
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
        
        # Dynamic range (using RMS frames with silence gating)
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
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0.0
        
        # Spectral rolloff (85%)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs[rolloff_idx[0]]
            else:
                spectral_rolloff = freqs[-1]
        else:
            spectral_rolloff = 0.0
        
        # Zero crossing rate
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
    
    def assess_quality(self, snr, lufs, stats):
        """Professional quality assessment with detailed scoring"""
        score = 0
        max_score = 100
        details = []
        
        # SNR scoring (30 points)
        if snr['global_snr'] >= 50:
            score += 30
            details.append("SNR: Excellent (50+ dB)")
        elif snr['global_snr'] >= 40:
            score += 25
            details.append("SNR: Very Good (40-50 dB)")
        elif snr['global_snr'] >= 30:
            score += 20
            details.append("SNR: Good (30-40 dB)")
        elif snr['global_snr'] >= 20:
            score += 10
            details.append("SNR: Acceptable (20-30 dB)")
        else:
            score += 5
            details.append("SNR: Poor (<20 dB)")
        
        # LUFS scoring (25 points)
        ideal_lufs = -16  # Streaming standard
        lufs_deviation = abs(lufs['integrated'] - ideal_lufs)
        
        if lufs_deviation <= 2:
            score += 25
            details.append("Loudness: Ideal for streaming")
        elif lufs_deviation <= 5:
            score += 20
            details.append("Loudness: Good")
        elif lufs_deviation <= 10:
            score += 15
            details.append("Loudness: Acceptable")
        elif lufs_deviation <= 15:
            score += 10
            details.append("Loudness: Suboptimal")
        else:
            score += 5
            details.append("Loudness: Poor")
        
        # Dynamic range scoring (20 points)
        if 15 <= stats['dynamic_range'] <= 30:
            score += 20
            details.append("Dynamic Range: Optimal")
        elif 10 <= stats['dynamic_range'] < 15 or 30 < stats['dynamic_range'] <= 40:
            score += 15
            details.append("Dynamic Range: Good")
        elif 5 <= stats['dynamic_range'] < 10 or 40 < stats['dynamic_range'] <= 50:
            score += 10
            details.append("Dynamic Range: Acceptable")
        else:
            score += 5
            details.append("Dynamic Range: Poor")
        
        # LRA scoring (15 points)
        if 7 <= lufs['lra'] <= 20:
            score += 15
            details.append("Loudness Range: Optimal")
        elif 5 <= lufs['lra'] < 7 or 20 < lufs['lra'] <= 25:
            score += 10
            details.append("Loudness Range: Good")
        else:
            score += 5
            details.append("Loudness Range: Poor")
        
        # True peak scoring (10 points)
        if lufs['true_peak_db'] <= -1:
            score += 10
            details.append("True Peak: Excellent headroom")
        elif lufs['true_peak_db'] <= -0.3:
            score += 7
            details.append("True Peak: Acceptable")
        else:
            score += 3
            details.append("True Peak: Risk of clipping")
        
        # Overall grade
        percentage = (score / max_score) * 100
        
        if percentage >= 85:
            grade = "Excellent - Broadcast/Master Quality"
            color = self.color_scheme['success']
        elif percentage >= 70:
            grade = "Good - Professional Quality"
            color = self.color_scheme['success']
        elif percentage >= 55:
            grade = "Acceptable - Consumer Quality"
            color = self.color_scheme['warning']
        elif percentage >= 40:
            grade = "Fair - Needs Improvement"
            color = self.color_scheme['warning']
        else:
            grade = "Poor - Significant Issues"
            color = self.color_scheme['danger']
        
        # Generate recommendations
        recommendations = []
        
        if snr['global_snr'] < 30:
            recommendations.append("• Apply noise reduction to improve SNR")
        
        if lufs_deviation > 5:
            if lufs['integrated'] < ideal_lufs:
                recommendations.append(f"• Increase level by {ideal_lufs - lufs['integrated']:.1f} dB for streaming")
            else:
                recommendations.append(f"• Decrease level by {lufs['integrated'] - ideal_lufs:.1f} dB for streaming")
        
        if lufs['lra'] < 5:
            recommendations.append("• Audio may be over-compressed, consider reducing compression")
        elif lufs['lra'] > 25:
            recommendations.append("• Very wide dynamic range, consider gentle compression")
        
        if lufs['true_peak_db'] > -0.3:
            recommendations.append("• Apply limiting to prevent clipping (target -1 dBTP)")
        
        if stats['dynamic_range'] < 10:
            recommendations.append("• Limited dynamic range detected, check compression settings")
        
        if not recommendations:
            recommendations.append("✓ Audio meets professional standards")
            recommendations.append("✓ Suitable for distribution")
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'grade': grade,
            'color': color,
            'details': details,
            'recommendations': recommendations
        }
    
    def create_professional_report(self):
        """Generate enhanced professional A4 PDF report"""
        print(f"  Generating professional report...")
        
        # Run all analyses
        snr = self.calculate_snr()
        lufs = self.calculate_lufs()
        bands, f, t, Sxx = self.analyze_frequency_bands()
        stats = self.calculate_statistics()
        quality = self.assess_quality(snr, lufs, stats)
        
        # Store results
        self.results = {
            'snr': snr,
            'lufs': lufs,
            'bands': bands,
            'stats': stats,
            'quality': quality,
            'metadata': self.metadata
        }
        
        # Professional A4 template
        figures = []
        
        # Page 1: Executive Summary
        fig1 = self.create_executive_summary_page()
        figures.append(fig1)
        
        # Page 2: Technical Analysis
        fig2 = self.create_technical_analysis_page()
        figures.append(fig2)
        
        # Page 3: Spectral Analysis
        fig3 = self.create_spectral_analysis_page()
        figures.append(fig3)
        
        # Page 4: Recommendations
        fig4 = self.create_recommendations_page()
        figures.append(fig4)
        
        return figures
    
    def create_executive_summary_page(self):
        """Create executive summary page with quality score and key metrics"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        # Header
        fig.text(0.5, 0.96, 'AudioQC ANALYSIS REPORT', 
                fontsize=18, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        fig.text(0.5, 0.94, 'Executive Summary', 
                fontsize=14, ha='center', color=self.color_scheme['secondary'])
        
        # Create grid with more rows for better spacing
        gs = GridSpec(7, 3, figure=fig, hspace=0.5, wspace=0.4,
                     top=0.90, bottom=0.05, left=0.08, right=0.92)
        
        # File information card
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        # Create info box with border
        info_rect = mpatches.FancyBboxPatch((0.02, 0.1), 0.96, 0.8,
                                           boxstyle="round,pad=0.02",
                                           facecolor='#f8f9fa',
                                           edgecolor=self.color_scheme['primary'],
                                           linewidth=2)
        ax_info.add_patch(info_rect)
        
        info_text = f"""
  File: {self.filename}
  Format: {self.metadata.get('format', 'Unknown')} | Channels: {self.metadata.get('channels', 1)} | Sample Rate: {self.sr} Hz
  Duration: {self.duration:.2f}s | Size: {self.file_size:.2f} MB | Bit Depth: {self.metadata.get('bit_depth', 'Unknown')}
        """
        ax_info.text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center',
                    transform=ax_info.transAxes)
        
        # Quality Score Gauge (spans 2 rows for prominence)
        ax_gauge = fig.add_subplot(gs[1:3, :])
        self.draw_professional_gauge(ax_gauge)
        
        # Key Metrics Cards - Fixed layout
        metrics = [
            ('SNR', f"{self.results['snr']['global_snr']:.1f} dB", self.get_metric_color('snr', self.results['snr']['global_snr'])),
            ('LUFS', f"{self.results['lufs']['integrated']:.1f}", self.get_metric_color('lufs', self.results['lufs']['integrated'])),
            ('LRA', f"{self.results['lufs']['lra']:.1f} LU", self.get_metric_color('lra', self.results['lufs']['lra'])),
            ('True Peak', f"{self.results['lufs']['true_peak_db']:.1f} dB", self.get_metric_color('peak', self.results['lufs']['true_peak_db'])),
            ('Dynamic Range', f"{self.results['stats']['dynamic_range']:.1f} dB", self.get_metric_color('dr', self.results['stats']['dynamic_range'])),
            ('Noise Floor', f"{self.results['snr']['noise_floor']:.1f} dB", self.color_scheme['neutral'])
        ]
        
        # Create two rows of 3 metrics each
        for i, (label, value, color) in enumerate(metrics):
            row = 3 if i < 3 else 4
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
            
            # Metric card with proper spacing
            rect = mpatches.FancyBboxPatch((0.1, 0.15), 0.8, 0.7,
                                          boxstyle="round,pad=0.02",
                                          facecolor=color, alpha=0.15,
                                          edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            
            # Value text (larger, positioned higher)
            ax.text(0.5, 0.6, value, fontsize=11, fontweight='bold',
                   ha='center', va='center', color=color,
                   transform=ax.transAxes)
            # Label text (smaller, positioned lower)
            ax.text(0.5, 0.25, label, fontsize=8, ha='center', va='center',
                   color=self.color_scheme['primary'],
                   transform=ax.transAxes)
        
        # Quick Assessment
        ax_assess = fig.add_subplot(gs[5:, :])
        ax_assess.axis('off')
        
        # Assessment box with improved spacing
        assess_rect = mpatches.FancyBboxPatch((0.02, 0.15), 0.96, 0.8,
                                             boxstyle="round,pad=0.02",
                                             facecolor=self.results['quality']['color'],
                                             alpha=0.1,
                                             edgecolor=self.results['quality']['color'],
                                             linewidth=2)
        ax_assess.add_patch(assess_rect)
        
        # Format text with proper line spacing
        assess_lines = []
        assess_lines.append(f"QUALITY ASSESSMENT: {self.results['quality']['grade']}")
        assess_lines.append("")
        assess_lines.append("KEY FINDINGS:")
        for detail in self.results['quality']['details'][:5]:
            assess_lines.append(f"  • {detail}")
        assess_lines.append("")
        assess_lines.append("TOP RECOMMENDATIONS:")
        for rec in self.results['quality']['recommendations'][:3]:
            assess_lines.append(f"  {rec}")
        
        assess_text = '\n'.join(assess_lines)
        
        ax_assess.text(0.05, 0.5, assess_text, fontsize=9,
                      transform=ax_assess.transAxes, va='center')
        
        # Footer
        fig.text(0.5, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | AudioQC Professional v0.1",
                fontsize=8, ha='center', color='gray')
        
        return fig
    
    def create_technical_analysis_page(self):
        """Create technical analysis page with waveform and loudness"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        # Header with proper spacing
        fig.text(0.5, 0.97, 'TECHNICAL ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3,
                     top=0.93, bottom=0.04, left=0.08, right=0.95)
        
        # Waveform with silence detection
        ax_wave = fig.add_subplot(gs[0, :])
        self.plot_waveform_professional(ax_wave)
        
        # RMS Energy
        ax_rms = fig.add_subplot(gs[1, :])
        self.plot_rms_energy_professional(ax_rms)
        
        # LUFS Timeline
        ax_lufs = fig.add_subplot(gs[2, :])
        self.plot_lufs_timeline_professional(ax_lufs)
        
        # Temporal SNR
        ax_temporal = fig.add_subplot(gs[3, :])
        self.plot_temporal_snr_professional(ax_temporal)
        
        # Statistics table
        ax_stats = fig.add_subplot(gs[4, :])
        self.create_statistics_table(ax_stats)
        
        return fig
    
    def create_spectral_analysis_page(self):
        """Create spectral analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'SPECTRAL ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.35,
                     top=0.93, bottom=0.04, left=0.08, right=0.95)
        
        # Spectrogram (spans full width)
        ax_spec = fig.add_subplot(gs[0:2, :])
        self.plot_spectrogram_professional(ax_spec)
        
        # Frequency bands SNR (full width)
        ax_bands = fig.add_subplot(gs[2, :])
        self.plot_frequency_bands_professional(ax_bands)
        
        # Amplitude distribution (left)
        ax_hist = fig.add_subplot(gs[3, 0])
        self.plot_amplitude_histogram(ax_hist)
        
        # LUFS distribution (right)
        ax_lufs_dist = fig.add_subplot(gs[3, 1])
        self.plot_lufs_distribution(ax_lufs_dist)
        
        return fig
    
    def create_recommendations_page(self):
        """Create detailed recommendations page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'PROFESSIONAL RECOMMENDATIONS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        # Main text area
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create professional recommendations text
        rec_text = self.generate_professional_recommendations()
        
        # Add text with proper formatting and spacing
        y_pos = 0.90  # Start position higher
        line_spacing = 0.022  # Consistent line spacing
        
        for section in rec_text:
            if section['type'] == 'header':
                ax.text(0.05, y_pos, section['text'], fontsize=12, fontweight='bold',
                       color=self.color_scheme['primary'], transform=ax.transAxes)
                y_pos -= line_spacing * 1.5
            elif section['type'] == 'subheader':
                ax.text(0.05, y_pos, section['text'], fontsize=10, fontweight='bold',
                       color=self.color_scheme['secondary'], transform=ax.transAxes)
                y_pos -= line_spacing * 1.3
            elif section['type'] == 'text':
                ax.text(0.07, y_pos, section['text'], fontsize=9,
                       color='black', transform=ax.transAxes)
                y_pos -= line_spacing
            elif section['type'] == 'separator':
                ax.plot([0.05, 0.95], [y_pos, y_pos], color='lightgray', linewidth=0.5,
                       transform=ax.transAxes)
                y_pos -= line_spacing * 0.8
        
        return fig
    
    # Helper plotting functions
    def draw_professional_gauge(self, ax):
        """Draw professional quality gauge with improved layout"""
        score = self.results['quality']['percentage']
        
        # Draw arc background
        theta = np.linspace(np.pi, 0, 100)
        r_outer = 1.0
        r_inner = 0.7
        
        # Color gradient based on score
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#27ae60']
        thresholds = [0, 40, 55, 70, 85, 100]
        
        for i in range(len(colors)):
            start = np.pi - (thresholds[i] / 100 * np.pi)
            end = np.pi - (thresholds[i + 1] / 100 * np.pi)
            theta_seg = np.linspace(start, end, 20)
            
            x_outer = r_outer * np.cos(theta_seg)
            y_outer = r_outer * np.sin(theta_seg)
            x_inner = r_inner * np.cos(theta_seg)
            y_inner = r_inner * np.sin(theta_seg)
            
            verts = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
            poly = mpatches.Polygon(verts, facecolor=colors[i], alpha=0.3, edgecolor=colors[i])
            ax.add_patch(poly)
        
        # Draw needle
        angle = np.pi - (score / 100 * np.pi)
        ax.arrow(0, 0, 0.85 * np.cos(angle), 0.85 * np.sin(angle),
                head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=2)
        
        # Center circle
        circle = plt.Circle((0, 0), 0.1, color='white', ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Score text - positioned lower to avoid overlap
        ax.text(0, -0.35, f"{score:.0f}%", fontsize=22, fontweight='bold',
               ha='center', va='center', color=self.results['quality']['color'])
        ax.text(0, -0.5, self.results['quality']['grade'], fontsize=10,
               ha='center', va='center', color=self.color_scheme['primary'])
        
        # Scale labels - positioned at the ends of the arc
        labels_pos = [(0, 'Poor'), (50, 'Fair'), (100, 'Excellent')]
        for val, label in labels_pos:
            angle = np.pi - (val / 100 * np.pi)
            x = 1.2 * np.cos(angle)
            y = 1.2 * np.sin(angle)
            # Adjust alignment based on position
            ha = 'center' if val == 50 else ('right' if val == 0 else 'left')
            ax.text(x, y, label, fontsize=9, ha=ha, va='center')
        
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.7, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Overall Quality Score', fontsize=11, fontweight='bold', pad=15)
    
    def get_metric_color(self, metric_type, value):
        """Get color based on metric value"""
        if metric_type == 'snr':
            if value >= 40:
                return self.color_scheme['success']
            elif value >= 30:
                return self.color_scheme['secondary']
            elif value >= 20:
                return self.color_scheme['warning']
            else:
                return self.color_scheme['danger']
        elif metric_type == 'lufs':
            deviation = abs(value + 16)  # Target -16 LUFS
            if deviation <= 2:
                return self.color_scheme['success']
            elif deviation <= 5:
                return self.color_scheme['secondary']
            else:
                return self.color_scheme['warning']
        elif metric_type == 'lra':
            if 7 <= value <= 20:
                return self.color_scheme['success']
            elif 5 <= value < 7 or 20 < value <= 25:
                return self.color_scheme['secondary']
            else:
                return self.color_scheme['warning']
        elif metric_type == 'peak':
            if value <= -1:
                return self.color_scheme['success']
            elif value <= -0.3:
                return self.color_scheme['warning']
            else:
                return self.color_scheme['danger']
        elif metric_type == 'dr':
            if 15 <= value <= 30:
                return self.color_scheme['success']
            elif 10 <= value < 15 or 30 < value <= 40:
                return self.color_scheme['secondary']
            else:
                return self.color_scheme['warning']
        else:
            return self.color_scheme['neutral']
    
    def plot_waveform_professional(self, ax):
        """Plot waveform with professional styling"""
        # Downsample for display
        display_samples = min(len(self.audio), 20000)
        if len(self.audio) > display_samples:
            indices = np.linspace(0, len(self.audio) - 1, display_samples, dtype=int)
            audio_display = self.audio[indices]
            time_display = indices / self.sr
        else:
            audio_display = self.audio
            time_display = np.arange(len(self.audio)) / self.sr
        
        # Plot waveform
        ax.fill_between(time_display, audio_display, alpha=0.5, color=self.color_scheme['secondary'])
        ax.plot(time_display, audio_display, linewidth=0.5, color=self.color_scheme['primary'])
        
        # Mark silence regions
        for start, end in self.results['snr']['silence_regions']:
            start_time = start * self.hop_length / self.sr
            end_time = end * self.hop_length / self.sr
            ax.axvspan(start_time, end_time, alpha=0.2, color=self.color_scheme['danger'])
        
        ax.set_xlabel('Time (s)', fontsize=9, labelpad=-1)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title(f'Waveform ({self.results["snr"]["silence_percentage"]:.1f}% silence)', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(-1.05, 1.05)
    
    def plot_rms_energy_professional(self, ax):
        """Plot RMS energy with professional styling"""
        rms_time = np.arange(len(self.results['snr']['rms'])) * self.hop_length / self.sr
        rms_db = 20 * np.log10(self.results['snr']['rms'] + 1e-10)
        
        ax.fill_between(rms_time, -60, rms_db, alpha=0.3, color=self.color_scheme['secondary'])
        ax.plot(rms_time, rms_db, linewidth=1, color=self.color_scheme['primary'])
        
        ax.axhline(y=self.results['snr']['noise_floor'], color=self.color_scheme['danger'],
                  linestyle='--', linewidth=1.5, alpha=0.8,
                  label=f'Noise: {self.results["snr"]["noise_floor"]:.1f} dB')
        ax.axhline(y=self.results['snr']['signal_level'], color=self.color_scheme['success'],
                  linestyle='--', linewidth=1.5, alpha=0.8,
                  label=f'Signal: {self.results["snr"]["signal_level"]:.1f} dB')
        
        ax.set_xlabel('Time (s)', fontsize=9, labelpad=-1)
        ax.set_ylabel('Level (dBFS)', fontsize=9)
        ax.set_title('RMS Energy Analysis', fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(max(-60, self.results['snr']['noise_floor'] - 10), 0)
    
    def plot_lufs_timeline_professional(self, ax):
        """Plot LUFS timeline with professional styling"""
        # Downsample if needed
        if len(self.results['lufs']['momentary']) > 500:
            step = len(self.results['lufs']['momentary']) // 500
            mom_display = self.results['lufs']['momentary'][::step]
            mom_times = self.results['lufs']['momentary_times'][::step]
        else:
            mom_display = self.results['lufs']['momentary']
            mom_times = self.results['lufs']['momentary_times']
        
        # Plot LUFS data
        ax.plot(mom_times, mom_display, linewidth=0.5, alpha=0.5,
               label='Momentary', color=self.color_scheme['neutral'])
        ax.plot(self.results['lufs']['short_term_times'], self.results['lufs']['short_term'],
               linewidth=1.5, label='Short-term', color=self.color_scheme['secondary'])
        ax.axhline(y=self.results['lufs']['integrated'], color=self.color_scheme['accent'],
                  linestyle='--', linewidth=2,
                  label=f'Integrated: {self.results["lufs"]["integrated"]:.1f} LUFS')
        
        # Target zones (no labels to avoid clutter)
        ax.axhspan(-18, -14, alpha=0.08, color='green')
        ax.axhspan(-24, -22, alpha=0.08, color='blue')
        
        # Add text annotations for targets instead of legend
        ax.text(self.duration * 0.98, -16, 'Streaming', fontsize=7, 
                ha='right', va='center', color='green', alpha=0.7)
        ax.text(self.duration * 0.98, -23, 'Broadcast', fontsize=7, 
                ha='right', va='center', color='blue', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=9, labelpad=-1)
        ax.set_ylabel('LUFS', fontsize=9)
        ax.set_title('Loudness Timeline (ITU-R BS.1770-4)', fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7, ncol=3, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, self.duration)
        ax.set_ylim(max(-60, self.results['lufs']['integrated'] - 20), 0)
    
    def plot_temporal_snr_professional(self, ax):
        """Plot temporal SNR variation with improved display"""
        if self.results['snr']['temporal_snr']:
            times = [(s['start'] + s['end'])/2 for s in self.results['snr']['temporal_snr']]
            values = [s['snr'] for s in self.results['snr']['temporal_snr']]
            
            # Quality zones (background)
            ax.axhspan(40, 100, alpha=0.08, color='green')
            ax.axhspan(30, 40, alpha=0.08, color='yellow')
            ax.axhspan(20, 30, alpha=0.08, color='orange')
            ax.axhspan(0, 20, alpha=0.08, color='red')
            
            # Plot data
            ax.plot(times, values, marker='o', linewidth=2, markersize=5,
                   color=self.color_scheme['primary'], label='Temporal SNR')
            ax.axhline(y=self.results['snr']['global_snr'], color=self.color_scheme['accent'],
                      linestyle='--', linewidth=1.5, alpha=0.8,
                      label=f'Avg: {self.results["snr"]["global_snr"]:.1f} dB')
            
            # Add quality zone labels on the right
            ax.text(self.duration * 1.01, 45, 'Excellent', fontsize=7, 
                   va='center', color='green', alpha=0.7)
            ax.text(self.duration * 1.01, 35, 'Good', fontsize=7, 
                   va='center', color='#d4a017', alpha=0.7)
            ax.text(self.duration * 1.01, 25, 'Fair', fontsize=7, 
                   va='center', color='orange', alpha=0.7)
            ax.text(self.duration * 1.01, 10, 'Poor', fontsize=7, 
                   va='center', color='red', alpha=0.7)
            
            ax.set_xlabel('Time (s)', fontsize=9, labelpad=-1)
            ax.set_ylabel('SNR (dB)', fontsize=9)
            ax.set_title('Temporal SNR Variation', fontsize=10, fontweight='bold')
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-self.duration * 0.02, self.duration * 1.02)
            ax.set_ylim(0, max(50, max(values) * 1.1))
    
    def plot_spectrogram_professional(self, ax):
        """Plot spectrogram with professional styling"""
        # Get spectrogram data
        _, f, t, Sxx = self.analyze_frequency_bands()
        
        # Limit frequency range
        max_freq_idx = np.where(f <= min(10000, self.sr/2))[0][-1]
        
        # Downsample for display
        if Sxx.shape[1] > 500:
            step = Sxx.shape[1] // 500
            Sxx_display = Sxx[:max_freq_idx, ::step]
            t_display = t[::step]
        else:
            Sxx_display = Sxx[:max_freq_idx, :]
            t_display = t
        
        # Plot
        im = ax.pcolormesh(t_display, f[:max_freq_idx],
                          10 * np.log10(Sxx_display + 1e-10),
                          shading='auto', cmap='viridis', rasterized=True)
        
        ax.set_ylabel('Frequency (Hz)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_title('Spectrogram (0-10kHz)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, min(10000, self.sr/2))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    
    def plot_frequency_bands_professional(self, ax):
        """Plot frequency bands SNR with improved layout"""
        bands = self.results['bands']
        
        names = [b['name'] for b in bands]
        values = [b['snr'] for b in bands]
        
        # Color based on SNR value
        colors = [self.get_metric_color('snr', v) for v in values]
        
        # Create bars with proper spacing
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            y_pos = height + max(values) * 0.02 if height > 0 else 0.5
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Set x-axis labels with rotation for better readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('SNR (dB)', fontsize=9)
        ax.set_title('Signal-to-Noise Ratio by Frequency Band', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Set y-axis limit with padding
        y_max = max(values) * 1.15 if values else 50
        ax.set_ylim(0, max(50, y_max))
        
        # Quality reference lines
        ax.axhline(y=40, color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=30, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=20, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    def plot_amplitude_histogram(self, ax):
        """Plot amplitude distribution with compact legend"""
        ax.hist(self.audio, bins=100, alpha=0.7, color=self.color_scheme['secondary'],
               edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Amplitude', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Amplitude Distribution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add statistics
        mean = np.mean(self.audio)
        std = np.std(self.audio)
        ax.axvline(x=mean, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=mean + std, color='orange', linestyle=':', linewidth=1, alpha=0.8)
        ax.axvline(x=mean - std, color='orange', linestyle=':', linewidth=1, alpha=0.8)
        
        # Compact text annotation instead of legend
        stats_text = f'μ={mean:.3f}\nσ={std:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_lufs_distribution(self, ax):
        """Plot LUFS distribution with compact display"""
        ax.hist(self.results['lufs']['momentary'], bins=50, alpha=0.7,
               color=self.color_scheme['accent'], edgecolor='black', linewidth=0.5)
        ax.axvline(x=self.results['lufs']['integrated'], color='red', linestyle='--',
                  linewidth=2, alpha=0.8)
        
        # Add text annotation instead of legend
        ax.text(self.results['lufs']['integrated'] + 1, ax.get_ylim()[1] * 0.9,
               f'Integrated:\n{self.results["lufs"]["integrated"]:.1f} LUFS',
               fontsize=8, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('LUFS', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Loudness Distribution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def create_statistics_table(self, ax):
        """Create statistics table with improved formatting"""
        ax.axis('off')

        # Create table data
        data = [
            ['Metric', 'Value', 'Target', 'Status'],
            ['Global SNR', f"{self.results['snr']['global_snr']:.1f} dB", '≥40 dB', self.get_status_symbol(self.results['snr']['global_snr'] >= 40)],
            ['Integrated LUFS', f"{self.results['lufs']['integrated']:.1f}", '-16±2', self.get_status_symbol(abs(self.results['lufs']['integrated'] + 16) <= 2)],
            ['True Peak', f"{self.results['lufs']['true_peak_db']:.1f} dB", '≤-1 dB', self.get_status_symbol(self.results['lufs']['true_peak_db'] <= -1)],
            ['LRA', f"{self.results['lufs']['lra']:.1f} LU", '7-20 LU', self.get_status_symbol(7 <= self.results['lufs']['lra'] <= 20)],
            ['Dynamic Range', f"{self.results['stats']['dynamic_range']:.1f} dB", '15-30 dB', self.get_status_symbol(15 <= self.results['stats']['dynamic_range'] <= 30)]
        ]

        # Table
        table = ax.table(cellText=data, loc='center', cellLoc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.15])

        table.auto_set_font_size(False)
        table.set_fontsize(9)         # +1 per leggibilità
        table.scale(1.25, 2.1)        # leggermente più grande/alta

        # Header row
        for j in range(4):
            cell = table[(0, j)]
            cell.set_facecolor(self.color_scheme['primary'])
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.10)     # prima 0.08

        # Data rows (tutte le righe sotto l'header)
        for i in range(1, len(data)):
            for j in range(4):
                cell = table[(i, j)]
                cell.set_height(0.15)  # prima 0.06
                if j == 3:  # status color
                    cell.set_facecolor('#d4edda' if '✓' in data[i][3] else '#f8d7da')

    
    def get_status_symbol(self, condition):
        """Get status symbol based on condition"""
        return '✓' if condition else '✗'
    
    def generate_professional_recommendations(self):
        """Generate structured recommendations"""
        sections = []
        
        # Header
        sections.append({'type': 'header', 'text': 'QUALITY ASSESSMENT SUMMARY'})
        sections.append({'type': 'text', 'text': f"Overall Score: {self.results['quality']['percentage']:.0f}% - {self.results['quality']['grade']}"})
        sections.append({'type': 'separator', 'text': ''})
        
        # Technical compliance
        sections.append({'type': 'subheader', 'text': 'TECHNICAL COMPLIANCE'})
        
        # Streaming standards
        sections.append({'type': 'text', 'text': '▸ Streaming Platforms (Target: -14 to -16 LUFS)'})
        lufs_val = self.results['lufs']['integrated']
        if -16 <= lufs_val <= -14:
            sections.append({'type': 'text', 'text': f'  ✓ Current: {lufs_val:.1f} LUFS - Optimal'})
        else:
            adjustment = -15 - lufs_val
            sections.append({'type': 'text', 'text': f'  ⚠ Current: {lufs_val:.1f} LUFS - Adjust by {adjustment:+.1f} dB'})
        
        # Broadcast standards
        sections.append({'type': 'text', 'text': '▸ Broadcast (EBU R128: -23 LUFS ±0.5)'})
        if -23.5 <= lufs_val <= -22.5:
            sections.append({'type': 'text', 'text': f'  ✓ Compliant'})
        else:
            sections.append({'type': 'text', 'text': f'  ✗ Non-compliant (adjust by {-23 - lufs_val:+.1f} dB)'})
        
        sections.append({'type': 'separator', 'text': ''})
        
        # Specific issues and solutions
        sections.append({'type': 'subheader', 'text': 'IDENTIFIED ISSUES & SOLUTIONS'})
        
        for rec in self.results['quality']['recommendations']:
            sections.append({'type': 'text', 'text': rec})
        
        sections.append({'type': 'separator', 'text': ''})
        
        # Processing suggestions
        sections.append({'type': 'subheader', 'text': 'RECOMMENDED PROCESSING CHAIN'})
        
        if self.results['snr']['global_snr'] < 30:
            sections.append({'type': 'text', 'text': '1. Noise Reduction (moderate to aggressive)'})
        
        if self.results['stats']['dynamic_range'] < 10:
            sections.append({'type': 'text', 'text': '2. Expansion (1:2 ratio, -40 dB threshold)'})
        elif self.results['stats']['dynamic_range'] > 40:
            sections.append({'type': 'text', 'text': '2. Compression (3:1 ratio, -20 dB threshold)'})
        
        if abs(lufs_val + 16) > 2:
            sections.append({'type': 'text', 'text': '3. Gain adjustment for target loudness'})
        
        if self.results['lufs']['true_peak_db'] > -1:
            sections.append({'type': 'text', 'text': '4. True Peak Limiting (ceiling: -1 dBTP)'})
        
        sections.append({'type': 'separator', 'text': ''})
        
        # Usage recommendations
        sections.append({'type': 'subheader', 'text': 'SUITABLE APPLICATIONS'})
        
        if self.results['quality']['percentage'] >= 70:
            sections.append({'type': 'text', 'text': '✓ Commercial release'})
            sections.append({'type': 'text', 'text': '✓ Streaming platforms'})
            sections.append({'type': 'text', 'text': '✓ Broadcast'})
        elif self.results['quality']['percentage'] >= 55:
            sections.append({'type': 'text', 'text': '✓ Podcast distribution'})
            sections.append({'type': 'text', 'text': '✓ Online content'})
            sections.append({'type': 'text', 'text': '⚠ Professional use (with processing)'})
        else:
            sections.append({'type': 'text', 'text': '⚠ Demo/reference only'})
            sections.append({'type': 'text', 'text': '✗ Not recommended for distribution'})
        
        return sections
    
    def save_report(self, output_path):
        """Save optimized PDF report"""
        figures = self.create_professional_report()
        
        with PdfPages(output_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, dpi=self.dpi, bbox_inches='tight',
                          pad_inches=0.3, facecolor='white')
                plt.close(fig)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = f'AudioQC Analysis - {self.filename}'
            d['Author'] = 'AudioQC Professional v0.1'
            d['Subject'] = 'Comprehensive Audio Quality Report'
            d['Keywords'] = 'Audio, Quality, SNR, LUFS, Analysis'
            d['CreationDate'] = datetime.now()
        
        # Report file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  ✓ Report saved: {output_path}")
        print(f"  ✓ Report size: {file_size_mb:.2f} MB")
        print(f"  ✓ Quality: {self.results['quality']['grade']}")
        print(f"  ✓ Score: {self.results['quality']['percentage']:.0f}%")
        
        return self.results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AudioQC Professional v0.1 - Enhanced Audio Quality Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats (with pydub): MP3, WAV, FLAC, M4A, OGG, AAC, WMA, and more
Native support: WAV (including ALAW/ULAW)

Examples:
  %(prog)s audio.wav
  %(prog)s podcast.mp3 -o reports/
  %(prog)s --install-deps

For full format support:
  pip install pydub
  brew install ffmpeg  # macOS
  apt install ffmpeg   # Linux
        """
    )
    
    parser.add_argument('input', nargs='?', help='Audio file to analyze')
    parser.add_argument('-o', '--output', default='audioqc_reports',
                       help='Output directory (default: audioqc_reports)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='PDF resolution in DPI (default: 100, lower=smaller file)')
    parser.add_argument('--install-deps', action='store_true',
                       help='Show installation commands for dependencies')
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("""
AudioQC Professional - Dependency Installation
===============================================

Required packages:
  pip install numpy scipy matplotlib

For universal format support:
  pip install pydub

For FFmpeg (required by pydub for non-WAV formats):
  macOS:    brew install ffmpeg
  Ubuntu:   sudo apt update && sudo apt install ffmpeg
  Windows:  Download from https://ffmpeg.org/download.html

Test installation:
  python -c "import pydub; print('✓ pydub installed')"
  ffmpeg -version
        """)
        return 0
    
    if not args.input:
        parser.print_help()
        return 1
    
    print("="*60)
    print("AudioQC Professional v0.1")
    print("Enhanced Audio Quality Analyzer")
    print("="*60)
    
    # Check dependencies
    if not PYDUB_AVAILABLE:
        print("\n⚠ Note: Install pydub for full format support")
        print("  pip install pydub")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process file
    try:
        filename = os.path.basename(args.input)
        name_only = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output, f"{name_only}_analysis.pdf")
        
        print(f"\nProcessing: {filename}")
        
        analyzer = ProfessionalAudioAnalyzer(args.input)
        analyzer.dpi = args.dpi
        
        results = analyzer.save_report(output_path)
        
        print(f"\n✓ Analysis complete!")
        print(f"✓ Professional report generated")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())