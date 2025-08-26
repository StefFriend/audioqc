"""
AudioQC GUI Module - PyQt5 interface for file/folder selection
"""

from version import __version__, __full_name__

import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QFileDialog, 
                             QCheckBox, QSpinBox, QProgressBar, QTextEdit,
                             QFrame, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon
import glob
from audioqc import AudioAnalyzer


class AnalysisWorker(QThread):
    """Worker thread for running audio analysis"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, output_dir, dpi=100):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.dpi = dpi
        self.results = {}
    
    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            if os.path.isfile(self.input_path):
                # Single file analysis
                self.analyze_file(self.input_path)
            elif os.path.isdir(self.input_path):
                # Directory analysis
                audio_files = []
                extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac', '*.ogg', '*.wma']
                
                for ext in extensions:
                    audio_files.extend(glob.glob(os.path.join(self.input_path, ext)))
                    audio_files.extend(glob.glob(os.path.join(self.input_path, ext.upper())))
                
                if not audio_files:
                    self.error.emit("No audio files found in the selected directory")
                    return
                
                self.progress.emit(f"Found {len(audio_files)} audio files")
                
                for i, file_path in enumerate(audio_files):
                    self.progress.emit(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
                    self.analyze_file(file_path)
            
            self.finished.emit(self.results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def analyze_file(self, file_path):
        try:
            filename = os.path.basename(file_path)
            name_only = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_dir, f"{name_only}_analysis.pdf")
            
            self.progress.emit(f"Analyzing: {filename}")
            
            analyzer = AudioAnalyzer(file_path)
            analyzer.dpi = self.dpi
            
            results = analyzer.save_report(output_path)
            self.results[file_path] = {
                'output_path': output_path,
                'results': results
            }
            
            self.progress.emit(f"✓ Completed: {filename}")
            
        except Exception as e:
            self.progress.emit(f"✗ Error processing {filename}: {str(e)}")


class AudioQCGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        
    def init_ui(self):
        self.setWindowTitle(f'{__full_name__} - Audio Quality Control')
        self.setGeometry(100, 100, 600, 500)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(__full_name__)
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel('Professional Audio Quality Control Tool')
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Input selection
        input_group = QGridLayout()
        
        # File selection
        input_group.addWidget(QLabel('Input:'), 0, 0)
        
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText('Select an audio file or folder containing audio files')
        input_group.addWidget(self.input_path, 0, 1)
        
        btn_layout = QHBoxLayout()
        self.btn_file = QPushButton('Browse File')
        self.btn_file.clicked.connect(self.browse_file)
        btn_layout.addWidget(self.btn_file)
        
        self.btn_folder = QPushButton('Browse Folder')
        self.btn_folder.clicked.connect(self.browse_folder)
        btn_layout.addWidget(self.btn_folder)
        
        input_group.addLayout(btn_layout, 1, 1)
        
        # Output directory
        input_group.addWidget(QLabel('Output Directory:'), 2, 0)
        
        self.output_path = QLineEdit()
        self.output_path.setText('audioqc_reports')
        self.output_path.setPlaceholderText('Directory where reports will be saved')
        input_group.addWidget(self.output_path, 2, 1)
        
        self.btn_output = QPushButton('Browse')
        self.btn_output.clicked.connect(self.browse_output)
        input_group.addWidget(self.btn_output, 3, 1)
        
        layout.addLayout(input_group)
        
        # Options
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel('DPI:'))
        self.dpi_spinbox = QSpinBox()
        self.dpi_spinbox.setMinimum(50)
        self.dpi_spinbox.setMaximum(300)
        self.dpi_spinbox.setValue(100)
        options_layout.addWidget(self.dpi_spinbox)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.btn_analyze = QPushButton('Start Analysis')
        self.btn_analyze.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.btn_analyze.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.btn_analyze)
        
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        button_layout.addWidget(self.btn_stop)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(150)
        self.log_output.setPlaceholderText('Analysis progress will be shown here...')
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Select Audio File',
            '',
            'Audio Files (*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma);;All Files (*)'
        )
        if file_path:
            self.input_path.setText(file_path)
    
    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            'Select Folder Containing Audio Files'
        )
        if folder_path:
            self.input_path.setText(folder_path)
    
    def browse_output(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            'Select Output Directory'
        )
        if folder_path:
            self.output_path.setText(folder_path)
    
    def start_analysis(self):
        input_path = self.input_path.text().strip()
        output_dir = self.output_path.text().strip()
        
        if not input_path:
            self.log_message("Please select an input file or folder")
            return
        
        if not os.path.exists(input_path):
            self.log_message(f"Input path does not exist: {input_path}")
            return
        
        if not output_dir:
            output_dir = 'audioqc_reports'
            self.output_path.setText(output_dir)
        
        # Disable controls
        self.btn_analyze.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_output.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear log
        self.log_output.clear()
        
        # Start worker thread
        self.worker = AnalysisWorker(input_path, output_dir, self.dpi_spinbox.value())
        self.worker.progress.connect(self.log_message)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
    
    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("Analysis stopped by user")
            self.reset_ui()
    
    def analysis_finished(self, results):
        self.log_message(f"Analysis completed! Processed {len(results)} files")
        self.reset_ui()
    
    def analysis_error(self, error_msg):
        self.log_message(f"Error: {error_msg}")
        self.reset_ui()
    
    def log_message(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
    
    def reset_ui(self):
        # Re-enable controls
        self.btn_analyze.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.btn_folder.setEnabled(True)
        self.btn_output.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)


def main():
    app = QApplication(sys.argv)
    window = AudioQCGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()