"""
AudioQC - Professional Audio Quality Control Tool

Modular audio analysis toolkit with SNR, LUFS, LNR, and STI measurements.
"""

from .version import __version__, __full_name__
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