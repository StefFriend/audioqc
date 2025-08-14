# AudioQC Professional

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)
[![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey.svg)](https://github.com/yourusername/foremhash)


*Comprehensive audio quality analyzer with verified SNR/LUFS calculations and professional PDF reports.*


---

## ✨ Features

* **Universal loader**

  * Native WAV (incl. A-law/μ-law via `wave`/`audioop`)
  * Optional **any format** (MP3, M4A, FLAC, OGG, AAC, WMA…) via `pydub` + **FFmpeg**
* **Robust SNR**

  * Adaptive silence detection, trimmed/median estimators, speech-weighted SNR (300–3400 Hz)
* **Standards-based Loudness (LUFS)**

  * ITU-R BS.1770-4 K-weighting, absolute/relative gating, momentary/short-term/integrated
* **True Peak estimate**

  * 4× oversampling true-peak (dBTP) check
* **Frequency analysis**

  * Spectrogram, band SNR (Sub-bass → Brilliance), amplitude & loudness distributions
* **Executive scoring**

  * Overall quality score + recommendations for streaming/broadcast/podcast targets
* **Professional PDF output**

  * 4 pages (A4): Executive Summary, Technical Analysis, Spectral Analysis, Recommendations
* **Practical ergonomics**

  * Auto mono conversion when multi-channel, metadata shown, adjustable DPI

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
```

**FFmpeg** (required by `pydub` for MP3/M4A/…):

* macOS: `brew install ffmpeg`
* Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
* Windows (Chocolatey): `choco install ffmpeg`
  or download from ffmpeg.org and add `ffmpeg` to PATH.

Handy helper in the tool:

```bash
python audioqc.py --install-deps
```

> 📦 (Optional) Create a `requirements.txt`
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

| Flag             |           Default | Description                                          |
| ---------------- | ----------------: | ---------------------------------------------------- |
| `input`          |                 — | Path to the audio file to analyze                    |
| `-o, --output`   | `audioqc_reports` | Output directory for the PDF report                  |
| `--dpi`          |             `100` | DPI for PDF figures (higher = sharper & larger file) |
| `--install-deps` |                 — | Prints platform-specific install tips                |

Output:

* A PDF report at: `audioqc_reports/<filename>_analysis.pdf`

---

## 🧾 Report Contents

**Page 1 – Executive Summary**

* File metadata (format, channels, SR, duration, size, bit depth)
* Overall quality **gauge** + grade
* Key metrics cards: Global SNR, LUFS (integrated), LRA, True Peak, Dynamic Range, Noise Floor
* Top findings & recommendations

**Page 2 – Technical Analysis**

* Waveform with **silence shading**
* RMS energy with **noise floor** / **signal level**
* Loudness timeline (momentary, short-term, integrated line)
* Temporal SNR plot
* Compact **statistics table**

**Page 3 – Spectral Analysis**

* Spectrogram (0–10 kHz)
* SNR by frequency band
* Amplitude & LUFS distributions

**Page 4 – Recommendations**

* Compliance vs. streaming/broadcast targets
* Issues & suggested processing chain
* Suitable applications

---

## 📐 Metrics (how they’re computed)

* **Global SNR (dB)**
  Adaptive silence detection → median noise RMS + trimmed mean signal RMS → `20·log10(signal/noise)`
* **Speech-weighted SNR**
  4th-order Butterworth band-pass 300–3400 Hz, then SNR as above
* **LUFS** (BS.1770-4)
  K-weighting (high-shelf + high-pass), 400 ms momentary, 3 s short-term, integrated with absolute (-70 LUFS) and relative (-10 LU) gating
* **True Peak (dBTP)**
  4× oversampled peak
* **Dynamic Range (dB)**
  Percentile-based range of **active** RMS frames (10th → 95th), gated by silence mask
* **LRA (LU)**
  95th – 10th percentile of short-term LUFS

> Note: True-peak estimation is approximate (oversampling FIR vs. full inter-sample peak detection); for mastering-grade TP, use a dedicated TP meter as well.

---

## 🎛️ Tips & Tuning

* **File size vs. quality**: Increase `--dpi` for sharper plots; decrease for smaller PDFs.
* **Silence detection**: See `detect_silence()` for threshold/duration settings.
* **Performance**: Very long files benefit from a lower DPI and/or reducing spectrogram resolution.
* **Layout**: The script saves with `bbox_inches='tight'` to compact margins.
  If legends or annotations fall outside axes in your environment, remove `bbox_inches='tight'` in `save_report()`.

---

## 🩺 Troubleshooting

* “**pydub not installed**” / “**Limited to WAV**”
  → `pip install pydub` and install FFmpeg (see install section).
* “**Could not load audio file**”
  → Check path & permissions; for exotic containers/codecs ensure FFmpeg decodes them.
* **Very high/low SNR**
  → Your content might be very quiet or very dynamic. Adjust silence parameters or verify the recording chain.
* **Clipped True Peak**
  → Mastering limiter suggested; target ≤ **-1 dBTP**.

---

## 📜 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
