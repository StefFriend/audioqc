"""
Report Generator Module for AudioQC
Handles PDF report generation with all analysis results
"""

from version import __version__, __full_name__
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.patches as mpatches
from datetime import datetime

class ReportGenerator:
    """Professional PDF report generator"""
    
    def __init__(self, data):
        """Initialize with analysis data"""
        self.filename = data['filename']
        self.filepath = data['filepath']
        self.file_hash = data['file_hash']
        self.file_size = data['file_size']
        self.duration = data['duration']
        self.sr = data['sr']
        self.audio = data['audio']
        self.frame_length = data['frame_length']
        self.hop_length = data['hop_length']
        self.results = data['results']
        self.metadata = data['metadata']
        self.dpi = data['dpi']
        
        # Store analysis date for consistent footer
        self.analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Color scheme
        self.color_scheme = {
            'primary': '#1e3a5f',      # Dark blue
            'secondary': '#4a90e2',    # Light blue
            'accent': '#f39c12',       # Orange
            'success': '#27ae60',      # Green
            'warning': '#e67e22',      # Orange
            'danger': '#e74c3c',       # Red
            'neutral': '#95a5a6'       # Gray
        }
    
    def add_footer(self, fig):
        """Add footer to any figure"""
        footer_text = f"{__full_name__} - Analysis Date: {self.analysis_date}"
        fig.text(0.5, 0.01, footer_text, fontsize=8, ha='center', va='center',
                color='gray', style='italic')
    
    def create_all_pages(self):
        """Create all report pages"""
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
        
        # Page 5: STI Analysis
        fig5 = self.create_sti_analysis_page()
        figures.append(fig5)
        
        # Page 6: Measurement Explanation
        fig6 = self.create_measurement_explanation_page()
        figures.append(fig6)
        
        # Page 7: Standards Reference
        fig7 = self.create_standards_reference_page()
        figures.append(fig7)
        
        return figures
    
    def save_pdf(self, out_path="AudioQC_Report.pdf"):
        # lock page size and prevent auto-cropping
        mpl.rcParams['savefig.bbox'] = 'standard'          # NOT 'tight'
        mpl.rcParams['figure.constrained_layout.use'] = False
        mpl.rcParams['savefig.pad_inches'] = 0.25

        figs = self.create_all_pages()
        with PdfPages(out_path) as pdf:
            for fig in figs:
                fig.set_size_inches(8.27, 11.69)           # A4 portrait
                pdf.savefig(fig, bbox_inches=None)         # keep full page
                plt.close(fig)

    def _fmt_row(self, label, value, unit="", vfmt="{:.1f}"):
        """Format a measurement row with proper alignment"""
        label_width = 34  # Width for label column
        value_width = 10  # Width for value column
        
        # Format the value
        formatted_value = vfmt.format(value)
        
        # Create the full value + unit string
        if unit:
            value_unit = f"{formatted_value} {unit}"
        else:
            value_unit = formatted_value
        
        # Pad label and right-align value+unit
        return f"  {label:<{label_width}}{value_unit:>{value_width+5}}"


    
    def create_executive_summary_page(self):
        """Create executive summary page with file hash and measurements"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        # Header
        fig.text(0.5, 0.96, 'AUDIOQC ANALYSIS REPORT', 
                fontsize=18, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        fig.text(0.5, 0.94, 'Technical Measurements', 
                fontsize=14, ha='center', color=self.color_scheme['secondary'])
        
        # Grid - adjusted for better centering (reduced from 6 to 5 since we remove STI quality)
        gs = GridSpec(5, 1, figure=fig, hspace=0.4,
                    top=0.89, bottom=0.07, left=0.1, right=0.9)
        
        # File information with hash
        ax_info = fig.add_subplot(gs[0])
        ax_info.axis('off')
        
        # Center the info box better
        info_rect = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.85,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#f8f9fa',
                                        edgecolor=self.color_scheme['primary'],
                                        linewidth=2)
        ax_info.add_patch(info_rect)
        
        info_text = f"""File: {self.filename}
    SHA256: {self.file_hash}
    Format: {self.metadata.get('format', 'Unknown')} | Channels: {self.metadata.get('channels', 1)}
    Sample Rate: {self.sr} Hz | Duration: {self.duration:.2f}s
    Size: {self.file_size:.2f} MB | Bit Depth: {self.metadata.get('bit_depth', 'Unknown')}"""
        
        ax_info.text(0.5, 0.5, info_text, fontsize=8.5, ha='center', va='center',
                    transform=ax_info.transAxes, family='monospace')
        
        # Key measurements table - more compact
        ax_table = fig.add_subplot(gs[1:3])
        ax_table.axis('off')
        
        # More compact measurements with STI properly included
        measurements_text = "\n".join([
            "SIGNAL METRICS",
            "────────────────────────────────────────────────────────────",
            self._fmt_row("Global SNR",              self.results['snr']['global_snr'],          "dB"),
            self._fmt_row("Speech-weighted SNR",     self.results['snr']['speech_weighted_snr'], "dB"),
            self._fmt_row("LNR (LUFS to Noise)",     self.results['lufs']['lnr'],                "LU"),
            self._fmt_row("STI",                     self.results['sti']['overall_sti'],         "",    vfmt="{:.3f}"),
            "",
            "LOUDNESS",
            "────────────────────────────────────────────────────────────",
            self._fmt_row("Integrated LUFS",         self.results['lufs']['integrated'],         "LUFS"),
            self._fmt_row("Max Momentary",           self.results['lufs']['max_momentary'],      "LUFS"),
            self._fmt_row("Max Short-term",          self.results['lufs']['max_short_term'],     "LUFS"),
            self._fmt_row("Loudness Range (LRA)",    self.results['lufs']['lra'],                "LU"),
            "",
            "LEVELS",
            "────────────────────────────────────────────────────────────",
            self._fmt_row("True Peak",               self.results['lufs']['true_peak_db'],       "dBTP"),
            self._fmt_row("Noise Floor (dB)",        self.results['snr']['noise_floor'],         "dBFS"),
            self._fmt_row("Noise Floor (LUFS)",      self.results['lufs']['noise_floor_lufs'],   "LUFS"),
            self._fmt_row("Signal Level",            self.results['snr']['signal_level'],        "dBFS"),
            "",
            "DYNAMICS",
            "────────────────────────────────────────────────────────────",
            self._fmt_row("Dynamic Range",           self.results['stats']['dynamic_range'],     "dB"),
            self._fmt_row("Crest Factor",            self.results['snr']['crest_factor'],        "dB"),
            self._fmt_row("Silence",                 self.results['snr']['silence_percentage'],  "%"),
        ])
        
        ax_table.text(
            0.06, 0.5, measurements_text,
            fontsize=8.5,
            family='DejaVu Sans Mono',   # reliable monospace
            transform=ax_table.transAxes,
            va='center',
            ha='left'
        )
        
        # Waveform preview - now takes the remaining space
        ax_wave = fig.add_subplot(gs[3:])
        self.plot_waveform_simple(ax_wave)
        
        # Add footer
        self.add_footer(fig)
        
        return fig
    
    def create_snr_lnr_page(self):
        """Create SNR and LNR analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'SNR AND LNR ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35,
                     top=0.89, bottom=0.07, left=0.1, right=0.9)
                
        # RMS Energy with noise floor
        ax_rms = fig.add_subplot(gs[0, :])
        self.plot_rms_energy(ax_rms)
        
        # Temporal SNR
        ax_snr = fig.add_subplot(gs[1, :])
        self.plot_temporal_snr(ax_snr)
        
        # Temporal LNR
        ax_lnr = fig.add_subplot(gs[2, :])
        self.plot_temporal_lnr(ax_lnr)
        
        # SNR vs LNR Comparison
        ax_compare = fig.add_subplot(gs[3, :])
        self.plot_snr_lnr_comparison(ax_compare)
        
        # Frequency bands SNR
        ax_bands = fig.add_subplot(gs[4, :])
        self.plot_frequency_bands(ax_bands)
        
        # Add footer
        self.add_footer(fig)
        
        return fig
    
    def create_lufs_analysis_page(self):
        """Create LUFS analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'LOUDNESS ANALYSIS (LUFS)', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.35,
                     top=0.93, bottom=0.05, left=0.08, right=0.95)
        
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
        
        # Add footer
        self.add_footer(fig)
        
        return fig
    
    def create_spectral_analysis_page(self):
        """Create spectral analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'SPECTRAL ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(3, 1, figure=fig, hspace=0.35,
                     top=0.93, bottom=0.05, left=0.08, right=0.95)
        
        # Spectrogram
        ax_spec = fig.add_subplot(gs[0:2])
        self.plot_spectrogram(ax_spec)
        
        # Spectral statistics
        ax_stats = fig.add_subplot(gs[2])
        self.plot_spectral_statistics(ax_stats)
        
        # Add footer
        self.add_footer(fig)
        
        return fig
    
    def create_sti_analysis_page(self):
        """Create STI analysis page"""
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        
        fig.text(0.5, 0.97, 'STI (SPEECH TRANSMISSION INDEX) ANALYSIS', 
                fontsize=16, fontweight='bold', ha='center', color=self.color_scheme['primary'])
        
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.35,
                     top=0.93, bottom=0.05, left=0.08, right=0.95)
        
        # STI over time
        ax_sti = fig.add_subplot(gs[0:2, :])
        self.plot_sti_timeline(ax_sti)
        
        # STI per frequency band
        ax_bands = fig.add_subplot(gs[2, :])
        self.plot_sti_bands(ax_bands)
        
        # STI distribution histogram
        ax_dist = fig.add_subplot(gs[3, 0])
        self.plot_sti_distribution(ax_dist)
        
        # STI statistics table
        ax_stats = fig.add_subplot(gs[3, 1])
        self.create_sti_statistics_table(ax_stats)
        
        # STI interpretation guide
        ax_guide = fig.add_subplot(gs[4, :])
        self.create_sti_interpretation_guide(ax_guide)
        
        # Add footer
        self.add_footer(fig)
        
        return fig
    
    def create_measurement_explanation_page(self):
        """Create page explaining SNR, LUFS, LNR, and STI"""
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

STI (Speech Transmission Index)
────────────────────────────────────────────────────────────────────────────────
STI quantifies speech intelligibility based on the modulation transfer function
approach. It analyzes how well amplitude modulations in speech are preserved
across octave bands from 125 Hz to 8 kHz.

• Measures modulation depth at 14 frequencies (0.63-12.5 Hz)
• Weighted across 7 octave bands for speech relevance
• Scale: 0.0 (unintelligible) to 1.0 (perfect)
• Standardized in IEC 60268-16

KEY DIFFERENCES
────────────────────────────────────────────────────────────────────────────────

SNR vs LNR vs STI:
• SNR: Linear amplitude domain, unweighted, technical quality
• LNR: Loudness domain, K-weighted for perception
• STI: Modulation domain, speech-specific intelligibility
• All assess clarity but from different perspectives
        """
        
        ax.text(0.05, 0.94, explanation_text, fontsize=8.5, family='monospace',
               transform=ax.transAxes, va='top')
        
        # Add footer
        self.add_footer(fig)
        
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

SPEECH INTELLIGIBILITY STANDARDS
────────────────────────────────────────────────────────────────────────────────

IEC 60268-16 (2020)
Sound system equipment - Part 16: Speech transmission index
• Defines STI calculation methodology
• Specifies octave band weights and modulation frequencies
• Validation procedures and measurement requirements

ISO 9921 (2003)
Ergonomics - Assessment of speech communication
• STI application guidelines
• Quality categories for different applications
• Environmental correction factors

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
• IEC 60268-16 for STI measurement
• ITU-T P.56 for speech-weighted measurements
• AES17-2020 for dynamic range assessment

Calibration: 0 dBFS = Full Scale Digital
Reference: 1 kHz sine wave at -20 dBFS
        """
        
        ax.text(0.05, 0.94, standards_text, fontsize=8, family='monospace',
               transform=ax.transAxes, va='top')
        
        # Add footer
        self.add_footer(fig)
        
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
        """Plot temporal LNR"""
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
        y_limit = max_value * 1.15
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
        
        max_value = max(values) if values else 1
        y_limit = max_value * 1.15
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
        f = self.results['spectral']['f']
        t = self.results['spectral']['t']
        Sxx = self.results['spectral']['Sxx']
        
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
    
    def plot_sti_timeline(self, ax):
        """Plot STI over time"""
        if 'temporal_sti' in self.results['sti'] and self.results['sti']['temporal_sti']:
            times = [s['time'] for s in self.results['sti']['temporal_sti']]
            values = [s['sti'] for s in self.results['sti']['temporal_sti']]
            
            # Step plot for STI values
            ax.step(times, values, where='post', linewidth=2, 
                   color=self.color_scheme['primary'], label='STI')
            
            # Average line
            ax.axhline(y=self.results['sti']['overall_sti'], 
                      color=self.color_scheme['accent'],
                      linestyle='--', linewidth=2, alpha=0.8,
                      label=f'Mean: {self.results["sti"]["overall_sti"]:.3f}')
            
            # Quality zones
            ax.axhspan(0.75, 1.0, alpha=0.1, color='green', label='Excellent')
            ax.axhspan(0.60, 0.75, alpha=0.1, color='yellow')
            ax.axhspan(0.45, 0.60, alpha=0.1, color='orange')
            ax.axhspan(0.30, 0.45, alpha=0.1, color='red')
            ax.axhspan(0.0, 0.30, alpha=0.1, color='darkred')
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('STI', fontsize=9)
            ax.set_title('Speech Transmission Index Over Time', fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, self.duration)
            ax.set_ylim(0, 1)
    
    def plot_sti_bands(self, ax):
        """Plot STI per frequency band"""
        if 'band_sti' in self.results['sti']:
            bands = list(self.results['sti']['band_sti'].keys())
            values = list(self.results['sti']['band_sti'].values())
            
            x_pos = np.arange(len(bands))
            bars = ax.bar(x_pos, values, color=self.color_scheme['secondary'], 
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{b} Hz' for b in bands], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('STI', fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_title('STI by Octave Band', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def plot_sti_distribution(self, ax):
        """Plot STI distribution histogram"""
        if 'sti_values' in self.results['sti']:
            ax.hist(self.results['sti']['sti_values'], bins=20, alpha=0.7,
                   color=self.color_scheme['accent'], edgecolor='black', linewidth=0.5)
            
            # Add mean line
            ax.axvline(x=self.results['sti']['overall_sti'], color='red', 
                      linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('STI', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title('STI Distribution', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_xlim(0, 1)
    
    def create_sti_statistics_table(self, ax):
        """Create STI statistics table"""
        ax.axis('off')
        
        stats_text = f"""
STI STATISTICS
──────────────────────
Mean STI:     {self.results['sti']['overall_sti']:.3f}
Min STI:      {self.results['sti']['sti_min']:.3f}
Max STI:      {self.results['sti']['sti_max']:.3f}
Std Dev:      {self.results['sti']['sti_std']:.3f}

Quality Rating:
{self.get_sti_quality_rating(self.results['sti']['overall_sti'])}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
               transform=ax.transAxes, va='center')
    
    def create_sti_interpretation_guide(self, ax):
        """Create STI interpretation guide"""
        ax.axis('off')
        
        guide_text = """
STI INTERPRETATION GUIDE
─────────────────────────────────────────────────────────────────────────────────────
STI Range    Quality        Intelligibility    Typical Application
─────────────────────────────────────────────────────────────────────────────────────
0.75 - 1.00  Excellent      Perfect           Recording studios, high-end conference rooms
0.60 - 0.75  Good           Clear speech      Good conference rooms, lecture halls
0.45 - 0.60  Fair           Acceptable        Average rooms, some background noise
0.30 - 0.45  Poor           Difficult         Noisy environments, poor acoustics
0.00 - 0.30  Bad            Unintelligible    Very noisy, severe acoustic problems
        """
        
        ax.text(0.05, 0.5, guide_text, fontsize=8, family='monospace',
               transform=ax.transAxes, va='center')
    
    def get_sti_quality_rating(self, sti_value):
        """Get quality rating based on STI value"""
        if sti_value >= 0.75:
            return "Excellent"
        elif sti_value >= 0.60:
            return "Good"
        elif sti_value >= 0.45:
            return "Fair"
        elif sti_value >= 0.30:
            return "Poor"
        else:
            return "Bad"