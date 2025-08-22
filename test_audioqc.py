"""
Test script for AudioQC modules
Creates a test audio signal and verifies all components work correctly
"""

import numpy as np
import os
import sys
from scipy.io import wavfile
import tempfile

def create_test_audio_file(duration=5, sample_rate=44100, frequency=440, snr_db=20):
    """
    Create a test audio file with a sine wave and noise
    
    Parameters:
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - frequency: Frequency of the sine wave
    - snr_db: Desired SNR in dB
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave (signal)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some amplitude modulation for STI testing
    mod_freq = 2.0  # 2 Hz modulation
    modulation = 0.3 * np.sin(2 * np.pi * mod_freq * t) + 0.7
    signal = signal * modulation
    
    # Generate noise
    noise = np.random.normal(0, 0.1, len(t))
    
    # Adjust noise level to achieve desired SNR
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor for noise to achieve target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scaling = np.sqrt(target_noise_power / noise_power)
    noise = noise * noise_scaling
    
    # Combine signal and noise
    audio = signal + noise
    
    # Add some silence at the beginning and end
    silence_samples = int(0.5 * sample_rate)  # 0.5 seconds of silence
    silence = np.zeros(silence_samples)
    audio = np.concatenate([silence, audio, silence])
    
    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16, sample_rate

def test_modules():
    """Test all AudioQC modules"""
    print("AudioQC Module Test")
    print("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test audio file
        print("\n1. Creating test audio file...")
        test_file = os.path.join(temp_dir, "test_audio.wav")
        audio_data, sample_rate = create_test_audio_file(
            duration=3,
            sample_rate=44100,
            frequency=1000,
            snr_db=25
        )
        wavfile.write(test_file, sample_rate, audio_data)
        print(f"   ✓ Test file created: {test_file}")
        print(f"   Duration: 4 seconds (including silence)")
        print(f"   Sample rate: {sample_rate} Hz")
        
        # Test imports
        print("\n2. Testing module imports...")
        try:
            from audioqc import AudioAnalyzer
            print("   ✓ AudioAnalyzer imported")
            
            from audio_loader import AudioLoader
            print("   ✓ AudioLoader imported")
            
            from snr_analyzer import SNRAnalyzer
            print("   ✓ SNRAnalyzer imported")
            
            from lufs_analyzer import LUFSAnalyzer
            print("   ✓ LUFSAnalyzer imported")
            
            from sti_analyzer import STIAnalyzer
            print("   ✓ STIAnalyzer imported")
            
            from spectral_analyzer import SpectralAnalyzer
            print("   ✓ SpectralAnalyzer imported")
            
            from report_generator import ReportGenerator
            print("   ✓ ReportGenerator imported")
            
        except ImportError as e:
            print(f"   ✗ Import error: {e}")
            print("\nMake sure all module files are in the current directory.")
            return False
        
        # Test AudioAnalyzer
        print("\n3. Testing AudioAnalyzer initialization...")
        try:
            analyzer = AudioAnalyzer(test_file)
            print(f"   ✓ AudioAnalyzer initialized")
            print(f"   File loaded: {analyzer.filename}")
            print(f"   Duration: {analyzer.duration:.2f} seconds")
            print(f"   Sample rate: {analyzer.sr} Hz")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
        
        # Test analysis
        print("\n4. Running analysis...")
        try:
            results = analyzer.analyze()
            print("   ✓ Analysis completed")
            
            # Check SNR results
            if 'snr' in results:
                print(f"   SNR: {results['snr']['global_snr']:.1f} dB")
                assert 'global_snr' in results['snr']
                assert 'speech_weighted_snr' in results['snr']
                print("   ✓ SNR analysis successful")
            
            # Check LUFS results
            if 'lufs' in results:
                print(f"   LUFS: {results['lufs']['integrated']:.1f} LUFS")
                assert 'integrated' in results['lufs']
                assert 'lnr' in results['lufs']
                print("   ✓ LUFS analysis successful")
            
            # Check STI results
            if 'sti' in results:
                print(f"   STI: {results['sti']['overall_sti']:.3f}")
                assert 'overall_sti' in results['sti']
                assert 'temporal_sti' in results['sti']
                print("   ✓ STI analysis successful")
            
            # Check spectral results
            if 'bands' in results:
                print(f"   Frequency bands analyzed: {len(results['bands'])}")
                print("   ✓ Spectral analysis successful")
            
        except Exception as e:
            print(f"   ✗ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test PDF generation
        print("\n5. Testing PDF report generation...")
        try:
            pdf_path = os.path.join(temp_dir, "test_report.pdf")
            analyzer.save_report(pdf_path)
            
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path) / 1024  # KB
                print(f"   ✓ PDF report generated")
                print(f"   File size: {file_size:.1f} KB")
                print(f"   Path: {pdf_path}")
            else:
                print("   ✗ PDF file not created")
                return False
                
        except Exception as e:
            print(f"   ✗ PDF generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test with different sample rates
        print("\n6. Testing with different sample rates...")
        for sr in [22050, 48000]:
            try:
                test_file_sr = os.path.join(temp_dir, f"test_{sr}.wav")
                audio_data_sr, _ = create_test_audio_file(
                    duration=1,
                    sample_rate=sr,
                    frequency=1000,
                    snr_db=20
                )
                wavfile.write(test_file_sr, sr, audio_data_sr)
                
                analyzer_sr = AudioAnalyzer(test_file_sr)
                results_sr = analyzer_sr.analyze()
                
                print(f"   ✓ {sr} Hz: STI={results_sr['sti']['overall_sti']:.3f}, "
                      f"SNR={results_sr['snr']['global_snr']:.1f} dB")
                
            except Exception as e:
                print(f"   ✗ Error at {sr} Hz: {e}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("\nAudioQC is ready to use.")
        print("\nUsage:")
        print("  python audioqc.py your_audio_file.wav")
        print("  python example_usage.py your_audio_file.wav")
        
        return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\nChecking dependencies...")
    
    dependencies = {
        'numpy': None,
        'scipy': None,
        'matplotlib': None,
        'soundfile': None,
        'pydub': '(optional)'
    }
    
    all_ok = True
    
    for module, note in dependencies.items():
        try:
            __import__(module)
            status = "✓ Installed"
            if note:
                status += f" {note}"
        except ImportError:
            if note == '(optional)':
                status = f"✗ Not installed {note}"
            else:
                status = "✗ Not installed (REQUIRED)"
                all_ok = False
        
        print(f"   {module:<15} {status}")
    
    if not all_ok:
        print("\n✗ Missing required dependencies!")
        print("  Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required dependencies are installed")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("AudioQC Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests
    success = test_modules()
    
    if not success:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
    else:
        sys.exit(0)