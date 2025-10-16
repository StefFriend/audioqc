"""
Spectral Analyzer Module for AudioQC
Handles frequency domain analysis
"""

import numpy as np
import scipy.signal

class SpectralAnalyzer:
    """Spectral and frequency band analyzer"""
    
    def __init__(self, audio, sr, frame_length):
        self.audio = audio
        self.sr = sr
        self.frame_length = frame_length
    
    def analyze_frequency_bands(self):
        """Analyze frequency content and band-specific SNR"""
        nperseg = min(self.frame_length, 2048)
        
        # Compute spectrogram locally for band analysis only
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
        
        # Return only compact summary; avoid carrying large Sxx in results
        return {
            'bands': band_analysis
        }
    
    def calculate_statistics(self, crest_factor_db):
        """Calculate comprehensive audio statistics"""
        eps = 1e-10
        
        # Basic statistics
        peak = np.max(np.abs(self.audio))
        peak_db = 20 * np.log10(peak + eps)
        
        rms = np.sqrt(np.mean(self.audio ** 2))
        rms_db = 20 * np.log10(rms + eps)
        
        # Dynamic range (simplified calculation)
        # Using frame-based RMS for more accurate dynamic range
        frame_length = self.frame_length
        hop_length = frame_length // 2
        
        n_frames = 1 + (len(self.audio) - frame_length) // hop_length
        rms_frames = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(self.audio))
            frame = self.audio[start:end]
            frame_rms = np.sqrt(np.mean(frame ** 2))
            if frame_rms > eps:
                rms_frames.append(frame_rms)
        
        if len(rms_frames) > 1:
            dr_high = np.percentile(rms_frames, 95)
            dr_low = np.percentile(rms_frames, 10)
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
            'zcr': float(zcr),
            'crest_factor': float(crest_factor_db)
        }
