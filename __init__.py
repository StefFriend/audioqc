"""
AudioQC - Professional Audio Quality Control Tool
Version 0.3

Modular audio analysis toolkit with SNR, LUFS, LNR, and STI measurements.
"""

__version__ = "0.3"
__author__ = "AudioQC"

# Import main components for package use
from .audioqc import AudioAnalyzer
from .audio_loader import AudioLoader
from .snr_analyzer import SNRAnalyzer
from .lufs_analyzer import LUFSAnalyzer
from .sti_analyzer import STIAnalyzer
from .spectral_analyzer import SpectralAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    'AudioAnalyzer',
    'AudioLoader',
    'SNRAnalyzer',
    'LUFSAnalyzer',
    'STIAnalyzer',
    'SpectralAnalyzer',
    'ReportGenerator'
]