# L-Avera-EFEST

Klasifikasi vokalisasi ayam petelur (*layer phase*) berbasis audio dan Machine Learning.
Fokus penelitian: optimasi arsitektur model untuk mempertahankan performa dengan kebutuhan
komputasi lebih rendah (target perangkat *edge/low-cost*).

---

## Dataset

**Poultry Vocalization Signal Dataset for Early Disease Detection**
- Kelas: `Healthy`, `Noise`, `Unhealthy`
- Lokasi: `Data/PoultryVocalization/Chicken_Audio_Dataset/`

---

## Arsitektur yang Diuji

| Model | Kode | Deskripsi |
|-------|------|-----------|
| Model A — Baseline CNN | `models/baseline_cnn.py` | CNN 1D filter 64→128→256 (referensi) |
| Model B — Compact CNN  | `models/compact_cnn.py`  | CNN 1D filter 32→64→128 |
| Model C — DS-CNN       | `models/ds_cnn.py`       | Depthwise Separable Conv1D |
| Model D — TC-ResNet8   | `models/tcresnet.py`     | Temporal Conv + Residual Shortcuts |

---

## Pipeline Preprocessing (bersama semua model)

```
WAV → resample 22 050 Hz → 40 MFCC → crop/pad 200 frames
Training: + white-noise augmentation (copy kedua per sampel)
Val/Test: tanpa augmentasi
```

Implementasi tunggal: `src/pipeline.py`

---

## Struktur Direktori

```
L-avera/
├── Data/                      # Dataset asli (tidak dimodifikasi)
├── models/                    # Definisi arsitektur
├── experiment/
│   ├── Baseline/              # Model A — baseline selesai
│   ├── CompactCNN/            # Model B
│   ├── DSCNN/                 # Model C
│   ├── TCResNet/              # Model D
│   └── int8/                  # Post-training quantization
├── artifacts/                 # Output per eksperimen
├── evaluation/                # Loader metrik & perbandingan
├── results/                   # Tabel & visualisasi final
└── src/                       # Shared pipeline
```

---

## Cara Menjalankan

Semua perintah dijalankan dari root direktori `L-avera/`.

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Baseline (sudah selesai)

```bash
python -m experiment.Baseline.audit_dataset
python -m experiment.Baseline.split_dataset
python -m experiment.Baseline.train
```

### 3. Eksperimen Model B, C, D

```bash
# Compact CNN
python -m experiment.CompactCNN.train

# DS-CNN
python -m experiment.DSCNN.train

# TC-ResNet8
python -m experiment.TCResNet.train
```


> Ketiga eksperimen menggunakan `artifacts/baseline/split_manifest.csv` yang sudah ada.
> Tidak perlu membuat split baru.

### 4. Perbandingan Model & Evaluasi

```bash
python -m evaluation.comparison
```

### 5. INT8 Quantization (TC-ResNet8 Terpilih)

Edit `experiment/int8/config.py`:
```python
CANDIDATE_MODEL_KEY  = "tcresnet"
CANDIDATE_MODEL_PATH = Path("artifacts/tcresnet/TCResNet8.keras")
```

```bash
python -m experiment.int8.quantize
python -m experiment.int8.evaluate
```

---

## Hasil Evaluasi & Perbandingan Model

### 1. Tabel Perbandingan Semua Model (Held-out Test Set)

> **Catatan**: Seluruh model dan versi kuantisasi INT8 dievaluasi pada test set yang sama (52 sampel, held-out). Model kandidat untuk INT8 adalah **TC-ResNet8** berdasarkan trade-off efisiensi komputasi dan performa deteksi ayam sakit (*Unhealthy Recall*) terbaik.

| Model | Arsitektur / Tipe | Status | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Unhealthy Recall | Parameter Total | Ukuran File |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A** | Baseline CNN (FP32) | **Selesai** | **96.15%** | **95.55%** | **96.73%** | **94.87%** | **100.00%** | ~886.979 | ~10.2 MB |
| **Model B** | Compact CNN (FP32) | **Selesai** | 88.46% | 85.55% | 91.84% | 84.62% | **100.00%** | ~413.027 | ~4.8 MB |
| **Model C** | DS-CNN (FP32) | **Selesai** | 61.54% | 48.62% | 51.28% | 53.70% | 61.11% | ~423.163 | ~4.9 MB |
| **Model D** | TC-ResNet8 (FP32) | **Selesai** | 94.23% | 93.24% | 95.35% | 92.31% | **100.00%** | ~65.723 | ~930 KB |
| **Model INT8** | TC-ResNet8 PTQ (INT8) | **Selesai** | 94.23% | 93.24% | 95.35% | 92.31% | **100.00%** | ~65.723 | **~95.5 KB** |

---

### 2. Rincian Metrik Model A — Baseline (Test Set: 52 Sampel)

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 95.45% | 100.00% | 97.67% | 21 |
| **Noise** | 100.00% | 84.62% | 91.67% | 13 |
| **Unhealthy** | 94.74% | 100.00% | 97.30% | 18 |
| **Rata-rata (Macro)** | **96.73%** | **94.87%** | **95.55%** | 52 |
| **Rata-rata (Weighted)** | 96.34% | 96.15% | 96.04% | 52 |

---

### 3. Rincian Metrik Model B — Compact CNN (Test Set: 52 Sampel)

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 80.77% | 100.00% | 89.36% | 21 |
| **Noise** | 100.00% | 53.85% | 70.00% | 13 |
| **Unhealthy** | 94.74% | 100.00% | 97.30% | 18 |
| **Rata-rata (Macro)** | **91.84%** | **84.62%** | **85.55%** | 52 |
| **Rata-rata (Weighted)** | 90.41% | 88.46% | 87.27% | 52 |

---

### 4. Rincian Metrik Model C — DS-CNN (Test Set: 52 Sampel)

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 53.85% | 100.00% | 70.00% | 21 |
| **Noise** | 0.00% | 0.00% | 0.00% | 13 |
| **Unhealthy** | 100.00% | 61.11% | 75.86% | 18 |
| **Rata-rata (Macro)** | **51.28%** | **53.70%** | **48.62%** | 52 |
| **Rata-rata (Weighted)** | 56.36% | 61.54% | 54.53% | 52 |

---

### 5. Rincian Metrik Model D — TC-ResNet8 (Test Set: 52 Sampel)

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 91.30% | 100.00% | 95.45% | 21 |
| **Noise** | 100.00% | 76.92% | 86.96% | 13 |
| **Unhealthy** | 94.74% | 100.00% | 97.30% | 18 |
| **Rata-rata (Macro)** | **95.35%** | **92.31%** | **93.24%** | 52 |
| **Rata-rata (Weighted)** | 94.67% | 94.23% | 93.97% | 52 |

---

### 6. Rincian Metrik Model INT8 — TC-ResNet8 PTQ (Test Set: 52 Sampel)

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 91.30% | 100.00% | 95.45% | 21 |
| **Noise** | 100.00% | 76.92% | 86.96% | 13 |
| **Unhealthy** | 94.74% | 100.00% | 97.30% | 18 |
| **Rata-rata (Macro)** | **95.35%** | **92.31%** | **93.24%** | 52 |
| **Rata-rata (Weighted)** | 94.67% | 94.23% | 93.97% | 52 |

---

### 7. Generate Laporan & Visualisasi Otomatis

Setelah menjalankan training untuk model lainnya, jalankan script evaluasi berikut:

```bash
python -m evaluation.comparison
```

Script ini akan secara otomatis membaca metrik dari setiap model dan menghasilkan:
- `results/matriks/comparison_table.csv` — Tabel rekapitulasi performa klasifikasi
- `results/effisiensi/efficiency_table.csv` — Parameter model & ukuran berkas
- `results/plot/macro_f1_comparison.png` — Grafik perbandingan Macro-F1
- `results/plot/accuracy_comparison.png` — Grafik perbandingan Akurasi
- `results/plot/unhealthy_recall_comparison.png` — Grafik perbandingan Recall Unhealthy

---

## Catatan Penelitian

- Split dilakukan di level file secara konsisten (`artifacts/baseline/split_manifest.csv`). Semua model diuji pada data pengujian yang sama persis tanpa *data leakage*.
- Test set **tidak boleh diubah** berdasarkan hasil eksperimen.
- Pemilihan kandidat model untuk kuantisasi INT8 diutamakan yang memiliki efisiensi komputasi tinggi dengan **Macro-F1** dan **Unhealthy Recall** terbaik (karena deteksi ayam sakit adalah prioritas kritis).

