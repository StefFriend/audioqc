"""
AudioQC v0.3 - Professional Audio Quality Control Tool
Main entry point and coordinator
"""

import numpy as np
import os
import sys
import argparse
import glob
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl


# Import modular components
from audio_loader import AudioLoader
from snr_analyzer import SNRAnalyzer
from lufs_analyzer import LUFSAnalyzer
from sti_analyzer import STIAnalyzer
from spectral_analyzer import SpectralAnalyzer
from report_generator import ReportGenerator

# Try to import pydub for universal audio format support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class AudioAnalyzer:
    """
    Professional audio analyzer with objective measurements
    Including SNR, LUFS, LNR, and STI
    """
    
    def __init__(self, audio_file, sr=None):
        """Initialize the analyzer"""
        self.loader = AudioLoader()
        
        if isinstance(audio_file, str):
            self.filename = os.path.basename(audio_file)
            self.filepath = audio_file
            print(f"  Loading: {self.filename}")
            
            # Load audio and metadata
            audio_data = self.loader.load_audio_universal(audio_file, sr)
            self.audio = audio_data['audio']
            self.sr = audio_data['sr']
            self.metadata = audio_data['metadata']
            self.file_hash = audio_data['file_hash']
            self.original_channels = audio_data['original_channels']
            self.file_size = audio_data['file_size']
            
            print(f"  SHA256: {self.file_hash[:16]}...")
            
            # Convert to mono if needed
            if len(self.audio.shape) > 1:
                self.audio = np.mean(self.audio, axis=0)
                print(f"  Converted from {self.original_channels} channels to mono")
        else:
            self.filename = "Audio Stream"
            self.filepath = None
            self.file_hash = "N/A"
            self.audio = audio_file
            self.sr = sr if sr else 44100
            self.original_channels = 1
            self.metadata = {}
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
        
        # Initialize analyzers
        self.snr_analyzer = SNRAnalyzer(self.audio, self.sr, self.frame_length, self.hop_length)
        self.lufs_analyzer = LUFSAnalyzer(self.audio, self.sr, self.hop_length)
        self.sti_analyzer = STIAnalyzer(self.audio, self.sr)
        self.spectral_analyzer = SpectralAnalyzer(self.audio, self.sr, self.frame_length)
        
        # Report generator
        self.report_generator = None
        
        # PDF settings
        self.dpi = 100
    
    def analyze(self):
        """Run all analyses"""
        print(f"  Running analysis...")
        
        # Run SNR analysis
        self.results['snr'] = self.snr_analyzer.calculate_snr()
        
        # Run LUFS and LNR analysis
        lufs_results = self.lufs_analyzer.calculate_lufs_and_lnr(
            self.results['snr']['silence_mask']
        )
        self.results['lufs'] = lufs_results
        
        # Run STI analysis
        self.results['sti'] = self.sti_analyzer.compute_sti()
        
        # Run spectral analysis
        bands_results = self.spectral_analyzer.analyze_frequency_bands()
        self.results['bands'] = bands_results['bands']
        self.results['spectral'] = bands_results
        
        # Calculate basic statistics
        self.results['stats'] = self.spectral_analyzer.calculate_statistics(
            self.results['snr']['crest_factor']
        )
        
        # Add metadata
        self.results['metadata'] = self.metadata
        self.results['file_hash'] = self.file_hash
        
        return self.results
    
    def create_report(self):
        """Generate professional PDF report"""
        print(f"  Generating analysis report...")
        
        # Run analysis if not done yet
        if not self.results:
            self.analyze()
        
        # Initialize report generator with all necessary data
        report_data = {
            'filename': self.filename,
            'filepath': self.filepath,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'duration': self.duration,
            'sr': self.sr,
            'audio': self.audio,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'results': self.results,
            'metadata': self.metadata,
            'dpi': self.dpi
        }
        
        self.report_generator = ReportGenerator(report_data)
        figures = self.report_generator.create_all_pages()
        
        return figures
    
    def save_report(self, output_path):
        """Save PDF report with fixed A4 page size (no auto-cropping)"""
        figures = self.create_report()

        # Prevent any global style from forcing tight cropping
        mpl.rcParams['savefig.bbox'] = 'standard'          # NOT 'tight'
        mpl.rcParams['figure.constrained_layout.use'] = False

        with PdfPages(output_path) as pdf:
            for fig in figures:
                # Ensure A4 portrait size is respected on every page
                fig.set_size_inches(8.27, 11.69)
                pdf.savefig(
                    fig,
                    dpi=self.dpi,
                    bbox_inches=None,                      # <-- key change
                    facecolor='white'
                    # pad_inches is ignored when bbox_inches=None, so omit it
                )
                plt.close(fig)

            # Metadata
            d = pdf.infodict()
            d['Title'] = f'Audio Analysis - {self.filename}'
            d['Author'] = 'AudioQC v0.3'
            d['Subject'] = 'AudioQC Analysis Report'
            d['Keywords'] = 'Audio, Analysis, SNR, LUFS, LNR, STI'
            d['CreationDate'] = datetime.now()

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Report saved: {output_path}")
        print(f"  ✓ Report size: {file_size_mb:.2f} MB")
        print(f"  ✓ Analysis complete")
        return self.results



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AudioQC v0.3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool provides objective audio measurements.

Measurements include:
  - SNR (Signal-to-Noise Ratio)
  - LUFS (Loudness Units relative to Full Scale) 
  - LNR (LUFS to Noise Ratio)
  - STI (Speech Transmission Index)
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
    print("AudioQC v0.3")
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