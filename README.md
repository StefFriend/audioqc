# AudioQC Professional

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)
[![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey.svg)](https://github.com/yourusername/foremhash)

*Comprehensive audio measurements (SNR/LUFS/LNR), with professional **6-page** PDF reports and file-integrity hash.*

---

## ✨ Features

* **Universal loader**
  * Native WAV (incl. **A-law/μ-law** via `wave`/`audioop`)
  * Optional **any format** (MP3, M4A, FLAC, OGG, AAC, WMA…) via `pydub` + **FFmpeg**
* **Robust SNR**
  * Adaptive silence detection, trimmed/median estimators, speech-weighted SNR (300–3400 Hz)
* **Standards-based Loudness (LUFS)**
  * ITU-R BS.1770-4 K-weighting, absolute/relative gating, momentary/short-term/integrated
* **LNR (LUFS-to-Noise Ratio) – NEW**
  * Computes average and **temporal LNR** (LU = loudness units), comparing K-weighted signal LUFS to noise LUFS
* **True Peak estimate**
  * 4× oversampling true-peak (dBTP) check
* **Frequency & spectral analysis**
  * Spectrogram; SNR by bands (Sub-bass → Brilliance); **spectral stats** (centroid, roll-off, ZCR)
* **Professional PDF output (A4, 6 pages)**
  1) Executive Summary (with **SHA256**),  
  2) **SNR & LNR** Analysis,  
  3) **Loudness (LUFS)**,  
  4) Spectral Analysis,  
  5) **Measurement Explanations**,  
  6) **Standards Reference**
* **Practical ergonomics**
  * Auto mono conversion for multi-channel, metadata display, adjustable DPI, clean layout

---

## 🔧 Installation

### Requirements

* Python **3.8+**
* Packages: `numpy`, `scipy`, `matplotlib`
  (optional for non-WAV formats: `pydub` and **FFmpeg**)

```bash
# minimal (WAV only)
pip install numpy scipy matplotlib

# full (all common formats)
pip install numpy scipy matplotlib pydub
````

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
> pydub  # optional
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

## 🧾 Report Contents

**Page 1 – Executive Summary**

* File metadata: format, channels, sample rate, duration, size, bit depth, **SHA256**
* Key measurements table:

  * Global SNR, **Speech-weighted SNR**, **LNR (LUFS-to-Noise)**, Integrated LUFS, Max Momentary, Max Short-term, **LRA**, **True Peak (dBTP)**, Noise Floor (dBFS & LUFS), Signal Level, **Dynamic Range**, **Crest Factor**, **Silence %**
* Waveform overview with **silence regions** shading
* Footer with analysis date & environment

**Page 2 – SNR & LNR Analysis (NEW)**

* RMS energy with **Noise Floor** / **Signal Level**
* **Temporal SNR** timeline
* **Temporal LNR** timeline (new metric)
* **SNR vs LNR** comparison chart
* SNR by frequency band

**Page 3 – Loudness (LUFS)**

* Loudness timeline (Momentary, Short-term, Integrated line)
* LUFS distribution & amplitude histogram
* **LUFS detailed statistics** table (Integrated, Max M/ST, LRA, Noise Floor LUFS, **LNR**, **True Peak**)

**Page 4 – Spectral Analysis**

* Spectrogram (up to Nyquist)
* **Spectral statistics**: centroid, roll-off (≈85%), zero-crossing rate, peak & RMS levels

**Page 5 – Measurement Explanations**

* Plain-English notes on **SNR**, **LUFS (BS.1770-4)**, **LNR** and an interpretation guide

**Page 6 – Technical Standards Reference**

* Summaries: ITU-R BS.1770-4, EBU R128, ATSC A/85, streaming targets, and measurement notes/calibration

---

## 📐 Metrics (how they’re computed)

* **Global SNR (dB)**
  Adaptive silence detection → noise RMS (median/percentile) & signal RMS (trimmed mean) → `20·log10(signal/noise)`
* **Speech-weighted SNR**
  4th-order Butterworth band-pass **300–3400 Hz**, then SNR as above
* **LUFS** (BS.1770-4)
  K-weighting; momentary (400 ms), short-term (3 s), integrated with absolute (-70 LUFS) and relative (-10 LU) gating
* **LNR (LUFS-to-Noise, LU)** – **NEW**
  Compute **signal LUFS** (K-weighted) and **noise LUFS** (K-weighted; from silent/low-energy portions);
  `LNR = signal_LUFS − noise_LUFS`. Also reported **temporal LNR** over sliding windows.
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

## 🎛️ Tips & Tuning

* **File size vs. quality**: Increase `--dpi` for sharper plots; decrease for smaller PDFs.
* **Silence detection**: Tunable percentile & minimum duration in `detect_silence()`.
* **Performance**: Very long files benefit from lower DPI and/or reduced spectrogram resolution.
* **Layout**: Reports save with `bbox_inches='tight'` (adjust in `save_report()` if legends/annotations clip).

---

## 🩺 Troubleshooting

* “**pydub not installed**” / “**Limited to WAV**” → `pip install pydub` and install FFmpeg.
* “**Could not load audio file**” → Check path & permissions; ensure FFmpeg decodes exotic codecs.
* **Very high/low SNR** → Content may be very quiet/dynamic; tweak silence parameters.
* **Clipped True Peak** → Use a limiter; target ≤ **-1 dBTP**.

---

## 📜 License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

```
```
