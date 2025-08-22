"""
SNR Analyzer Module for AudioQC
Handles Signal-to-Noise Ratio calculations
"""

import numpy as np
import scipy.signal

class SNRAnalyzer:
    """SNR (Signal-to-Noise Ratio) analyzer"""
    
    def __init__(self, audio, sr, frame_length, hop_length):
        self.audio = audio
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
    
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
                    'end': min((i + 1) * window_duration, len(self.audio) / self.sr),
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
            'noise_rms': noise_rms
        }