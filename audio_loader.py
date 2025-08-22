"""
Audio Loader Module for AudioQC
Handles universal audio format loading
"""

import numpy as np
import os
import wave
import audioop
import hashlib

# NEW: prefer soundfile like the reference script
try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

from scipy.io import wavfile
from scipy.signal import resample_poly

# pydub for broad format support (ffmpeg)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _to_float64_mono(x):
    """Ensure mono float64 in [-1, 1]."""
    x = np.asarray(x)
    if x.ndim > 1:
        # assume (samples, channels) or (channels, samples); make it (samples, channels)
        if x.shape[0] < x.shape[-1] and x.ndim == 2:
            # likely (channels, samples) -> transpose to (samples, channels)
            x = x.T
        x = x.mean(axis=1)
    return x.astype(np.float64, copy=False)


def _resample_if_needed(x, sr, target_sr):
    """Polyphase resampling if target_sr set and != sr."""
    if target_sr is None or target_sr == sr or x.size == 0:
        return x, sr
    # Use rational approximation
    from math import gcd
    g = gcd(int(target_sr), int(sr))
    up = int(target_sr // g)
    down = int(sr // g)
    y = resample_poly(x, up, down)
    return y, int(target_sr)


class AudioLoader:
    """Universal audio loader with metadata extraction"""

    def __init__(self):
        self.pydub_available = PYDUB_AVAILABLE
        self.sf_available = SF_AVAILABLE

    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of the file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def load_audio_universal(self, filepath, target_sr=None):
        """
        Universal audio loader with metadata extraction

        Returns dict with:
          - audio (mono float64)
          - sr
          - metadata {format, original_sr, bit_depth, channels, codec}
          - file_hash
          - original_channels
          - file_size (MB)
        """
        result = {
            'audio': None,
            'sr': None,
            'metadata': {
                'format': os.path.splitext(filepath)[1][1:].upper(),
                'original_sr': None,
                'bit_depth': None,
                'channels': None,
                'codec': None
            },
            'file_hash': self.calculate_file_hash(filepath),
            'original_channels': 1,
            'file_size': os.path.getsize(filepath) / (1024*1024) if os.path.exists(filepath) else 0.0
        }

        # ---- Method 1: soundfile first (matches your other module) ----
        if self.sf_available:
            try:
                data, fs = sf.read(filepath, always_2d=False)
                # soundfile returns float in [-1,1] or int; handle channels
                orig_channels = data.shape[1] if (isinstance(data, np.ndarray) and data.ndim == 2) else 1
                result['metadata']['original_sr'] = fs
                result['metadata']['channels'] = orig_channels
                result['metadata']['bit_depth'] = None  # sf doesn't expose easily
                result['metadata']['codec'] = 'PCM/Other (via soundfile)'
                result['original_channels'] = orig_channels

                audio = _to_float64_mono(data)
                audio, fs2 = _resample_if_needed(audio, fs, target_sr)
                result['audio'] = audio
                result['sr'] = fs2
                return result
            except Exception as e:
                print(f"  soundfile failed: {e}. Falling back...")

        # ---- Method 2: pydub (robust for compressed formats) ----
        if self.pydub_available:
            try:
                print("  Attempting to load with pydub/ffmpeg...")
                seg = AudioSegment.from_file(filepath)
                # Record original metadata
                result['metadata']['original_sr'] = seg.frame_rate
                result['metadata']['bit_depth'] = seg.sample_width * 8
                result['metadata']['channels'] = seg.channels
                result['metadata']['codec'] = 'via ffmpeg/pydub'
                result['original_channels'] = seg.channels

                # Standardize: force mono and 16-bit PCM for clean NumPy conversion
                seg = seg.set_channels(1)
                seg = seg.set_sample_width(2)  # int16
                sr = seg.frame_rate

                # IMPORTANT: do NOT use seg.set_frame_rate() for resampling; we resample ourselves
                # Extract raw samples (int16 little-endian)
                audio_i16 = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float64) / 32768.0

                audio, fs2 = _resample_if_needed(audio_i16, sr, target_sr)
                result['audio'] = audio
                result['sr'] = fs2
                print("  Successfully loaded with pydub.")
                return result
            except Exception as e:
                print(f"  pydub failed: {e}. Trying WAV/scipy fallbacks...")

        # ---- Method 3: wave module for WAV (handles ALAW/ULAW correctly) ----
        if filepath.lower().endswith('.wav'):
            try:
                with wave.open(filepath, 'rb') as w:
                    params = w.getparams()
                    sr = params.framerate
                    n_channels = params.nchannels
                    n_frames = params.nframes
                    sampwidth = params.sampwidth
                    comptype = params.comptype

                    result['metadata']['original_sr'] = sr
                    result['metadata']['bit_depth'] = sampwidth * 8
                    result['metadata']['channels'] = n_channels
                    result['metadata']['codec'] = comptype
                    result['original_channels'] = n_channels

                    frames = w.readframes(n_frames)

                    if comptype == 'NONE':
                        if sampwidth == 1:
                            # unsigned 8-bit -> [-1,1]
                            x = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
                            x = (x - 128.0) / 128.0
                        elif sampwidth == 2:
                            x = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
                        elif sampwidth == 4:
                            x = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
                        else:
                            # uncommon widths -> fallback normalize
                            x = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
                    elif comptype.upper() == 'ALAW':
                        decoded = audioop.alaw2lin(frames, 2)
                        x = np.frombuffer(decoded, dtype=np.int16).astype(np.float64) / 32768.0
                    elif comptype.upper() == 'ULAW':
                        decoded = audioop.ulaw2lin(frames, 2)
                        x = np.frombuffer(decoded, dtype=np.int16).astype(np.float64) / 32768.0
                    else:
                        # Unknown compression -> try int16 decode as best effort
                        x = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0

                    if n_channels > 1:
                        x = x.reshape(-1, n_channels)
                    audio = _to_float64_mono(x)
                    audio, fs2 = _resample_if_needed(audio, sr, target_sr)

                    result['audio'] = audio
                    result['sr'] = fs2
                    return result
            except Exception as e:
                print(f"  Wave module error: {e}")

        # ---- Method 4: scipy.io.wavfile fallback ----
        try:
            sr, x = wavfile.read(filepath)
            result['metadata']['original_sr'] = sr
            # Normalize by dtype
            if x.dtype == np.uint8:
                x = (x.astype(np.float64) - 128.0) / 128.0
                result['metadata']['bit_depth'] = 8
            elif x.dtype == np.int16:
                x = x.astype(np.float64) / 32768.0
                result['metadata']['bit_depth'] = 16
            elif x.dtype == np.int32:
                x = x.astype(np.float64) / 2147483648.0
                result['metadata']['bit_depth'] = 32
            elif x.dtype in (np.float32, np.float64):
                x = x.astype(np.float64, copy=False)
                result['metadata']['bit_depth'] = 32 if x.dtype == np.float32 else 64
            else:
                x = x.astype(np.float64)
                # scale to [-1,1] if needed
                maxv = np.max(np.abs(x)) or 1.0
                x = x / maxv

            if x.ndim > 1:
                result['metadata']['channels'] = x.shape[1]
                result['original_channels'] = x.shape[1]
            else:
                result['metadata']['channels'] = 1

            audio = _to_float64_mono(x)
            audio, fs2 = _resample_if_needed(audio, sr, target_sr)
            result['audio'] = audio
            result['sr'] = fs2
            return result

        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {e}")
