# AudioQC Professional

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)
[![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey.svg)](https://github.com/yourusername/foremhash)

*Comprehensive audio measurements (SNR/LUFS/LNR/STI), with professional **7-page** PDF reports and file-integrity hash.*

---

## ✨ Features

* **Universal loader**

  * Native WAV (incl. **A-law/μ-law** via `wave`/`audioop`)
  * Optional **any format** (MP3, M4A, FLAC, OGG, AAC, WMA…) via `pydub` + **FFmpeg**
* **Robust SNR**

  * Adaptive silence detection, trimmed/median estimators, speech-weighted SNR (300–3400 Hz)
* **Standards-based Loudness (LUFS)**

  * ITU-R BS.1770-4 K-weighting, absolute/relative gating, momentary/short-term/integrated
* **LNR (LUFS-to-Noise Ratio)**

  * Computes average and **temporal LNR** (LU), comparing K-weighted signal LUFS to noise LUFS
* **STI (Speech Transmission Index)**

  * IEC 60268-16 modulation-transfer based **speech intelligibility** (0.00–1.00), temporal trend and per-octave band stats
  * Implementation inspired by: Costantini, G., Paoloni, A., & Todisco, M. (2010), *Objective Speech Intelligibility Measures Based on Speech Transmission Index for Forensic Applications*, AES 39th Int. Conf., Hillerød, Denmark
* **True Peak estimate**

  * 4× oversampling true-peak (dBTP) check
* **Frequency & spectral analysis**

  * Spectrogram; SNR by bands (Sub-bass → Brilliance); **spectral stats** (centroid, roll-off, ZCR)
* **Professional PDF output (A4, 7 pages)**

  1. Executive Summary (with **SHA256**)
  2. **SNR & LNR** Analysis
  3. **Loudness (LUFS)**
  4. Spectral Analysis
  5. **STI Analysis** *(dedicated page)*
  6. **Measurement Explanations**
  7. **Technical Standards Reference**
* **Practical ergonomics**

  * Auto mono conversion for multi-channel, metadata display, adjustable DPI, clean layout

---

## 🔧 Installation

### Requirements

* Python **3.8+**
* Packages: `numpy`, `scipy`, `matplotlib`
  (optional for extended formats: `pydub`, **FFmpeg**; recommended: `soundfile`)

```bash
# minimal (WAV only)
pip install numpy scipy matplotlib

# full (all common formats)
pip install numpy scipy matplotlib pydub soundfile
```

**FFmpeg** (required by `pydub` for MP3/M4A/…):

* macOS: `brew install ffmpeg`
* Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
* Windows (Chocolatey): `choco install ffmpeg`
  or download from ffmpeg.org and add `ffmpeg` to PATH.

> 📦 (Optional) `requirements.txt`
>
> ```
> numpy
> scipy
> matplotlib
> soundfile  # recommended
> pydub      # optional (requires FFmpeg)
> ```

---

## ▶️ Usage

Basic:

```bash
python audioqc.py <input-audio-file>
```

Specify output folder & DPI:

```bash
python audioqc.py podcast.mp3 -o reports/ --dpi 120
```

Command-line options:

| Flag           |           Default | Description                                          |
| -------------- | ----------------: | ---------------------------------------------------- |
| `input`        |                 — | Path to the audio file to analyze                    |
| `-o, --output` | `audioqc_reports` | Output directory for the PDF report                  |
| `--dpi`        |             `100` | DPI for PDF figures (higher = sharper & larger file) |

Output:

* A PDF report at: `audioqc_reports/<filename>_analysis.pdf`
  (Path, file size, and basic info are printed in the console. The PDF embeds metadata:
  Title/Author/Subject/Keywords.)

---

## 🗂 Project Structure

```
audioqc/
│
├── audioqc.py              # Main entry point
├── audio_loader.py         # Universal audio loading (WAV + optional FFmpeg)
├── snr_analyzer.py         # SNR analysis (global & speech-weighted)
├── lufs_analyzer.py        # LUFS + LNR (BS.1770-4)
├── sti_analyzer.py         # STI analysis (IEC 60268-16)
├── spectral_analyzer.py    # Spectral analysis & stats
├── report_generator.py     # 7-page PDF report builder
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔊 Supported Audio Formats

**Basic install:**

* WAV (PCM, A-law, μ-law)

**With `pydub` + FFmpeg:**

* MP3, M4A/AAC, OGG/Vorbis, FLAC, WMA, AIFF, **and more**

---

## 🧾 Report Contents

**Page 1 – Executive Summary**

* File metadata: format, channels, sample rate, duration, size, bit depth, **SHA256**

* Key measurements table:

  * Global SNR, **Speech-weighted SNR**, **LNR (LUFS-to-Noise)**, Integrated LUFS, Max Momentary, Max Short-term, **LRA**, **True Peak (dBTP)**, Noise Floor (dBFS & LUFS), Signal Level, **Dynamic Range**, **Crest Factor**, **Silence %**

* Waveform overview with **silence regions** shading

* Footer with analysis date & environment

**Page 2 – SNR & LNR Analysis**

* RMS energy with **Noise Floor** / **Signal Level**
* **Temporal SNR** and **Temporal LNR** timelines
* SNR by frequency band

**Page 3 – Loudness (LUFS)**

* Loudness timeline (Momentary, Short-term, Integrated line)
* LUFS distribution & amplitude histogram
* **LUFS detailed statistics** table (Integrated, Max M/ST, LRA, Noise Floor LUFS, **LNR**, **True Peak**)

**Page 4 – Spectral Analysis**

* Spectrogram (up to Nyquist)
* **Spectral statistics**: centroid, roll-off (≈85%), zero-crossing rate, peak & RMS levels

**Page 5 – STI Analysis**

* **STI over time** with mean line and quality bands
* **STI by octave band** (125–8k Hz)
* **STI distribution** with mean marker and **STI statistics** panel
* **STI interpretation guide** (Excellent → Bad ranges)

**Page 6 – Measurement Explanations**

* Plain-English notes on **SNR**, **LUFS (BS.1770-4)**, **LNR**, **STI** and an interpretation guide

**Page 7 – Technical Standards Reference**

* Matches the section below.

---

## 📐 Metrics (how they’re computed)

* **Global SNR (dB)**
  Adaptive silence detection → noise RMS (median/percentile) & signal RMS (trimmed mean) → `20·log10(signal/noise)`

* **Speech-weighted SNR**
  4th-order Butterworth band-pass **300–3400 Hz**, then SNR as above

* **LUFS** (BS.1770-4)
  K-weighting; momentary (400 ms), short-term (3 s), integrated with absolute (−70 LUFS) and relative (−10 LU) gating

* **LNR (LUFS-to-Noise, LU)**
  Compute **signal LUFS** (K-weighted) and **noise LUFS** (K-weighted; from silent/low-energy portions);
  `LNR = signal_LUFS − noise_LUFS`. Also reported **temporal LNR** over sliding windows.

* **STI (Speech Transmission Index)**
  IEC 60268-16 modulation-transfer approach across octave bands (125–8k Hz) and 14 modulation freqs (0.63–12.5 Hz).
  Output 0.00–1.00 with temporal curve and band means.
  *Implementation inspired by Costantini, Paoloni & Todisco (AES 39th, 2010).*

* **True Peak (dBTP)**
  4× oversampled peak amplitude → `20·log10(|peak|)`

* **Dynamic Range (dB)**
  Percentile range of **active** RMS frames (10th → 95th), gated by silence mask

* **LRA (LU)**
  95th − 10th percentile of short-term LUFS

* **Spectral stats**
  Spectral centroid & roll-off (≈85%), zero-crossing rate

> Note: True-peak is an estimate; for mastering-grade TP use a dedicated meter as well.

---

## 🧭 Technical Standards Reference (README aligned with report)

### Loudness Standards

* **ITU-R BS.1770-4 (2015)**
  Algorithms to measure audio programme loudness and true-peak audio level
  • Defines K-weighting filters for loudness measurement
  • Specifies gating algorithm for integrated loudness
  • True-peak measurement via oversampling
* **EBU R128 (2020)**
  Loudness normalisation and permitted maximum level of audio signals
  • Target Level: −23.0 LUFS (±0.5 LU tolerance)
  • Max True Peak: −1 dBTP
  • Loudness Range: 5–20 LU typical

### Speech Intelligibility Standards

* **IEC 60268-16 (2020)**
  Sound system equipment – Part 16: Speech transmission index
  • Defines STI calculation methodology
  • Specifies octave band weights and modulation frequencies
  • Validation procedures and measurement requirements
* **ISO 9921 (2003)**
  Ergonomics – Assessment of speech communication
  • STI application guidelines
  • Quality categories for different applications
  • Environmental correction factors

### Signal Quality Specifications

* **ITU-T P.56 (2011)**
  Objective measurement of active speech level
  • Defines speech-band filtering (300–3400 Hz)
  • Active speech level measurement methodology
* **AES17-2020**
  Standard method for digital audio engineering measurement
  • Dynamic range measurement procedures
  • Signal-to-noise ratio definitions
  • Measurement bandwidth specifications

### Streaming Platform Specifications

* **Spotify (2021):** −14 LUFS integrated, −1 dBTP max
* **Apple Music (2021):** −16 LUFS integrated, −1 dBTP max
* **YouTube (2021):** −14 LUFS integrated
* **Amazon Music (2021):** −14 to −9 LUFS integrated
* **Tidal (2021):** −14 LUFS integrated

### Measurement Notes

* All measurements in this report comply with:
  • **ITU-R BS.1770-4** for LUFS calculation
  • **IEC 60268-16** for STI measurement
  • **ITU-T P.56** for speech-weighted measurements
  • **AES17-2020** for dynamic range assessment
* **Calibration:** 0 dBFS = Full Scale Digital
* **Reference:** 1 kHz sine wave at −20 dBFS

---

## 🎛️ Tips & Tuning

* **File size vs. quality**: Increase `--dpi` for sharper plots; decrease for smaller PDFs.
* **Silence detection**: Tunable percentile & minimum duration in `detect_silence()`.
* **Performance**: Very long files benefit from lower DPI and/or reduced spectrogram resolution.
* **Layout**: Reports save with `bbox_inches='tight'` (adjust in `save_report()` if legends/annotations clip).

---

## 🩺 Troubleshooting

* “**pydub not installed**” / “**Limited to WAV**” → `pip install pydub` and install FFmpeg.
* “**Could not load audio file**” → Check path & permissions; ensure FFmpeg decodes exotic codecs.
* **Very high/low SNR/LNR/STI** → Content may be very quiet/dynamic; tweak silence parameters.
* **Clipped True Peak** → Use a limiter; target ≤ **−1 dBTP**.
* **PDF issues** → Ensure `matplotlib` is up to date: `pip install --upgrade matplotlib`

---

## 🕰 Version History

* **v0.3**: Added **dedicated STI page** and expanded to **7-page** PDF; modular architecture
* **v0.2**: Added **LNR** metric, improved SNR calculation
* **v0.1**: Initial release with basic SNR and LUFS

---

## 📜 License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For issues or suggestions, please open an issue in this repository.
