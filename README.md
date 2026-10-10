# L-Avera-EFEST
### Lightweight Audio-Based Vocalization Classification for Early Screening of Laying Hens

L-Avera-EFEST merupakan proyek penelitian yang berfokus pada klasifikasi suara ayam petelur menggunakan *Machine Learning* dengan mempertimbangkan efisiensi model untuk perangkat berkomputasi terbatas. Penelitian ini mengevaluasi beberapa arsitektur neural network untuk mengetahui trade-off antara performa klasifikasi, jumlah parameter, dan ukuran model.

Proyek ini diarahkan sebagai langkah awal pengembangan sistem pemantauan akustik yang berpotensi mendukung identifikasi awal pola vokalisasi ayam yang tidak biasa. Sistem tidak ditujukan untuk menggantikan pemeriksaan dokter hewan dan tidak digunakan sebagai alat diagnosis penyakit klinis.

## 1. Latar Belakang

Pemantauan kondisi ayam petelur merupakan salah satu aspek penting dalam pengelolaan peternakan. Suara ayam berpotensi menyediakan informasi tambahan mengenai kondisi lingkungan dan perubahan perilaku vokalisasi. Namun, penerapan sistem klasifikasi berbasis audio pada perangkat dengan sumber daya terbatas memerlukan model yang tidak hanya memiliki performa memadai, tetapi juga efisien dalam penggunaan komputasi dan penyimpanan.

Model dengan arsitektur besar dapat menghasilkan performa klasifikasi yang baik, tetapi belum tentu ideal untuk penerapan pada perangkat *edge*. Oleh karena itu, L-Avera-EFEST mengevaluasi beberapa arsitektur dengan tingkat kompleksitas berbeda dan menguji model TC-ResNet8 dalam format kuantisasi INT8.

Fokus utama penelitian ini adalah mengevaluasi sejauh mana kompleksitas model dapat dikurangi tanpa kehilangan performa klasifikasi yang berarti, terutama dalam mengenali sampel dengan label **Unhealthy**.

## 2. Tujuan Penelitian

Penelitian ini memiliki beberapa tujuan:

1. Mengembangkan pipeline klasifikasi suara ayam petelur menggunakan fitur audio MFCC.
2. Membandingkan performa beberapa arsitektur neural network untuk klasifikasi tiga kelas.
3. Mengevaluasi trade-off antara metrik klasifikasi, jumlah parameter, dan ukuran berkas model.
4. Menguji penerapan *post-training quantization* INT8 pada TC-ResNet8 untuk mengurangi ukuran model.
5. Mengidentifikasi arsitektur yang berpotensi dikembangkan lebih lanjut untuk sistem klasifikasi audio berbasis *edge computing*.

## 3. Dataset

Penelitian menggunakan dataset **Poultry Vocalization Signal Dataset for Early Disease Detection** yang tersedia di Mendeley Data.

- Sumber: [Mendeley Data](https://data.mendeley.com/datasets/zp4nf2dxbh/1)
- DOI: [10.17632/zp4nf2dxbh.1](https://doi.org/10.17632/zp4nf2dxbh.1)
- Jumlah berkas audio: 346 berkas WAV.

Distribusi kelas pada dataset:

| Kelas | Jumlah |
|---|---:|
| Healthy | 139 |
| Noise | 86 |
| Unhealthy | 121 |
| **Total** | **346** |

Label `Healthy`, `Noise`, dan `Unhealthy` digunakan sesuai pengelompokan dataset. Label `Unhealthy` tidak secara otomatis menunjukkan diagnosis penyakit tertentu.

### Pembagian Dataset

Dataset dibagi menggunakan pendekatan stratified file-level split dengan proporsi:

- Training: 70%
- Validation: 15%
- Testing: 15%

Pembagian menggunakan seed 42 dan disimpan dalam `artifacts/baseline/split_manifest.csv` agar eksperimen antarmodel menggunakan pembagian data yang konsisten.

Pengujian yang dilaporkan menggunakan **52 berkas audio**, dengan distribusi:

| Kelas | Sampel uji |
|---|---:|
| Healthy | 21 |
| Noise | 13 |
| Unhealthy | 18 |
| **Total** | **52** |

Augmentasi white noise diterapkan hanya pada data training. Data validation dan testing tidak diberi augmentasi.

**Catatan metodologis:** pembagian dilakukan pada tingkat berkas. Identitas ayam, sesi perekaman, atau sumber rekaman belum dipastikan terpisah antarsplit. Oleh sebab itu, hasil belum dapat dianggap sebagai bukti generalisasi terhadap ayam atau lingkungan perekaman yang sepenuhnya berbeda.

## 4. Preprocessing Audio

Audio diproses menjadi fitur Mel-Frequency Cepstral Coefficients (MFCC) sebelum dimasukkan ke model.

Tahapan preprocessing:

1. Membaca berkas audio WAV.
2. Mengubah audio menjadi mono.
3. Melakukan resampling ke 22.050 Hz.
4. Mengekstraksi 40 koefisien MFCC.
5. Melakukan crop atau padding hingga 200 frame.
6. Menghasilkan representasi fitur dengan bentuk `(200, 40)`.

Representasi tersebut digunakan sebagai masukan model untuk mempelajari pola temporal dan karakteristik spektral dari vokalisasi ayam.

Implementasi preprocessing bersama untuk seluruh model (termasuk Baseline yang telah disinkronkan) menggunakan modul tunggal `src/pipeline.py`.

## 5. Arsitektur Model

Empat arsitektur utama dievaluasi dalam eksperimen ini.

| Model | Karakteristik |
|---|---|
| Baseline CNN | Model pembanding dengan kanal konvolusi 64 → 128 → 256 |
| Compact CNN | CNN yang lebih kecil dengan kanal 32 → 64 → 128 |
| DS-CNN | Menggunakan depthwise separable convolution |
| TC-ResNet8 | Arsitektur temporal convolution dengan residual connection |

Selain itu, TC-ResNet8 dievaluasi kembali setelah *post-training quantization* INT8.

### Mengapa TC-ResNet8?

TC-ResNet8 dipertimbangkan karena hasil eksperimen menunjukkan bahwa model ini memiliki jumlah parameter dan ukuran berkas yang jauh lebih kecil daripada Baseline CNN, dengan penurunan performa klasifikasi yang relatif terbatas pada pengujian saat ini.

Pemilihan ini didasarkan pada trade-off antara performa dan efisiensi, bukan karena TC-ResNet8 memiliki akurasi tertinggi pada seluruh eksperimen.

## 6. Hasil Eksperimen

### 6.1 Perbandingan Model

Hasil evaluasi pada test set:

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Recall Unhealthy |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 96,15% | 96,73% | 94,87% | 95,55% | 100,00% |
| Compact CNN | 88,46% | 91,84% | 84,62% | 85,55% | 100,00% |
| DS-CNN | 61,54% | 51,28% | 53,70% | 48,62% | 61,11% |
| TC-ResNet8 | 94,23% | 95,35% | 92,31% | 93,24% | 100,00% |
| TC-ResNet8 INT8 | 94,23% | 95,35% | 92,31% | 93,24% | 100,00% |

### 6.2 Kompleksitas dan Ukuran Model

| Model | Jumlah Parameter | Beban MACCs | MFLOPs | Ukuran Berkas Model |
|---|---:|---:|---:|---:|
| Baseline CNN | 886.979 | 9.180.544 | 18,36 M | ±10,2 MB |
| Compact CNN | 413.027 | 2.864.000 | 5,73 M | ±4,8 MB |
| DS-CNN | 423.163 | 1.323.584 | 2,65 M | ±4,9 MB |
| TC-ResNet8 | 65.723 | 3.033.744 | 6,07 M | ±930 KB |
| TC-ResNet8 INT8 | 65.723 (origin) | 3.033.744 (INT8) | 6,07 M | ±95,5 KB |

Ukuran berkas merupakan ukuran artefak model yang tercatat dan bukan ukuran keseluruhan aplikasi beserta runtime, preprocessing, atau kebutuhan memori saat inferensi. Jumlah parameter pada baris INT8 mengacu pada model asal sebelum kuantisasi, bukan jumlah parameter berpresisi integer yang dihitung ulang.

*Metodologi Perhitungan MACCs & FLOPs:* Beban komputasi dihitung secara deterministik untuk pemrosesan 1 sampel berdimensi `(1, 200, 40)`. Untuk lapisan Conv1D, $\text{MACCs} = L_{\text{out}} \times K \times C_{\text{in}} \times C_{\text{out}}$; untuk SeparableConv1D, dihitung akumulasi depthwise ($L_{\text{out}} \times K \times C_{\text{in}}$) dan pointwise ($L_{\text{out}} \times C_{\text{in}} \times C_{\text{out}}$); serta lapisan Dense dihitung sebagai $N_{\text{in}} \times N_{\text{out}}$. Konvensi standar $1 \text{ MACC} \approx 2 \text{ FLOPs}$ (1 operasi perkalian + 1 operasi akumulasi) digunakan untuk menghitung FLOPs.

Dibandingkan Baseline CNN, TC-ResNet8 memiliki sekitar 92,6% lebih sedikit parameter dan memangkas 67,0% operasi perkalian-akumulasi (MACCs). Akurasi TC-ResNet8 lebih rendah 1,92 poin persentase, sedangkan Macro-F1 lebih rendah 2,31 poin persentase.

Ukuran berkas TC-ResNet8 INT8 tercatat sekitar 89,7% lebih kecil dibandingkan model TC-ResNet8 berformat float32.

### 6.3 Interpretasi Hasil

**Baseline CNN**

Baseline CNN memperoleh performa klasifikasi tertinggi pada pengujian ini, dengan akurasi 96,15% dan Macro-F1 95,55%. Namun, jumlah parameter dan ukuran berkasnya relatif besar dibandingkan TC-ResNet8.

**Compact CNN**

Compact CNN mengurangi kompleksitas model dibandingkan Baseline CNN, tetapi Macro-F1 turun menjadi 85,55%. Model masih memperoleh Recall Unhealthy sebesar 100% pada test set, tetapi kemampuan mengenali kelas Noise lebih rendah.

**DS-CNN**

DS-CNN memperoleh akurasi 61,54% dan Macro-F1 48,62%. Recall untuk kelas Noise adalah 0% pada test set, sehingga model belum menunjukkan performa yang memadai untuk klasifikasi tiga kelas dalam konfigurasi eksperimen saat ini.

**TC-ResNet8**

TC-ResNet8 memperoleh akurasi 94,23% dan Macro-F1 93,24% dengan 65.723 parameter dan 3,03M MACCs. Hasil tersebut menunjukkan potensi sebagai kompromi optimal antara performa klasifikasi, efisiensi parameter, dan kecepatan komputasi.

**TC-ResNet8 INT8**

Evaluasi empiris melalui audit artefak independen mengonfirmasi bahwa model TFLite INT8 menghasilkan label prediksi yang 100% konsisten (52/52 sampel uji cocok sempurna) dengan versi float32. Kuantisasi bobot ke 8-bit integer hanya menghasilkan pergeseran probabilitas rata-rata yang sangat kecil (Mean Absolute Error = 0,0043), sehingga urutan argmax prediksi kelas tidak berubah pada seluruh sampel pengujian saat ini.

### 6.4 Perhatian terhadap Recall Unhealthy

Recall Unhealthy sebesar 100% pada Baseline CNN, Compact CNN, dan TC-ResNet8 berarti seluruh 18 sampel berlabel Unhealthy dalam test set dikenali dengan benar pada eksperimen tersebut.

Hasil ini menjanjikan, tetapi jumlah sampel uji masih terbatas. Nilai tersebut tidak membuktikan bahwa sistem akan selalu mengenali seluruh kondisi tidak sehat pada data baru dan tidak dapat diartikan sebagai sensitivitas klinis terhadap penyakit tertentu.

## 7. Kuantisasi dan Potensi Edge Computing

Kuantisasi INT8 diterapkan pada TC-ResNet8 untuk mengurangi ukuran artefak model. Tujuannya adalah menurunkan kebutuhan penyimpanan model dan mengeksplorasi kemungkinan penerapan pada perangkat dengan sumber daya terbatas.

Tahapan implementasi yang tersedia meliputi:

1. Memuat model TC-ResNet8 yang telah dilatih.
2. Melakukan konversi model melalui TensorFlow Lite dengan kalibrasi representatif.
3. Menggunakan konfigurasi kuantisasi integer penuh (`OpsSet.TFLITE_BUILTINS_INT8`, input/output int8).
4. Mengevaluasi model TFLite menggunakan interpreter pada test set.
5. Mengekspor model TFLite ke format C-array header (`artifacts/int8/model_data.h`).

### 7.1 Estimasi Alokasi Memori ESP32-S3
Berdasarkan analisis arsitektur mikrokontroler target (ESP32-S3 Xtensa LX7 dual-core):
- **Alokasi Flash:** Bobot model INT8 (95,45 KB) dialokasikan pada Flash ROM (`.rodata`), tidak membebani RAM internal saat standby.
- **Kebutuhan SRAM Runtime (Tensor Arena):** Analisis siklus hidup tensor (*tensor lifetime analysis*) menunjukkan puncak aktivasi simultan (*peak live activations*) sebesar **11.200 bytes (~10,94 KB)** saat transisi input (`1×200×40`) ke stem conv (`1×200×16`). Dengan memperhitungkan metadata struktur `TfLiteEvalTensor` (~1,4 KB), scratch buffer kernel konvolusi (~4 KB), dan padding alignment 16-byte, kebutuhan minimum arena adalah **~18,0 KB**. Alokasi Tensor Arena sebesar **~32 KB** memberikan margin headroom keamanan yang sangat memadai.
- **Kelayakan Perangkat:** ESP32-S3 memiliki 512 KB Internal SRAM (~350 KB usable heap). Kebutuhan ~32 KB hanya menyerap **~9,1% dari usable heap internal**, sehingga model **layak berjalan langsung pada SRAM internal tanpa memerlukan modul eksternal PSRAM**.
- **Kesiapan Integrasi:** Berkas C-array 16-byte aligned (`artifacts/int8/model_data.h`) telah diverifikasi identik secara biner (SHA-256 terverifikasi) untuk integrasi firmware dengan TensorFlow Lite for Microcontrollers (TFLite Micro) dan akselerasi instruksi vektor ESP-NN.

**Batasan:** Hasil ini merupakan analisis teoritis berbasis tensor lifetime dan ekspor artefak firmware; pengujian fisik langsung (waktu inferensi nyata dalam ms, profiling heap langsung di hardware, dan konsumsi daya aktual pada board ESP32-S3) masih merupakan langkah pengujian selanjutnya.

## 8. Struktur Repositori

```text
L-Avera/
├── models/
│   ├── baseline_cnn.py
│   ├── compact_cnn.py
│   ├── ds_cnn.py
│   └── tcresnet.py
│
├── experiment/
│   ├── Baseline/
│   │   ├── audit_dataset.py
│   │   ├── config.py
│   │   ├── split_dataset.py
│   │   └── train.py
│   ├── CompactCNN/
│   │   ├── config.py
│   │   └── train.py
│   ├── DSCNN/
│   │   ├── config.py
│   │   └── train.py
│   ├── TCResNet/
│   │   ├── config.py
│   │   └── train.py
│   └── int8/
│       ├── config.py
│       ├── quantize.py
│       └── evaluate.py
│
├── src/
│   └── pipeline.py
│
├── evaluation/
│   ├── metrics.py
│   ├── efficiency.py
│   └── comparison.py
│
├── artifacts/
│   ├── baseline/
│   ├── compact_cnn/
│   ├── ds_cnn/
│   ├── tcresnet/
│   └── int8/
│
└── README.md
```

Struktur di atas merupakan gambaran direktori kode dan artefak eksperimen. Dataset audio dan model terlatih tidak harus tersedia di repositori publik, sehingga pengguna perlu memastikan berkas yang diperlukan telah disiapkan sebelum menjalankan eksperimen.

## 9. Instalasi dan Persiapan

### 9.1 Clone Repositori

```bash
git clone https://github.com/hafidzjnr/L-Avera.git
cd L-Avera
```

### 9.2 Buat Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux atau macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 9.3 Instal Dependensi

Jika repositori menyediakan `requirements.txt`, jalankan:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika berkas tersebut belum tersedia, dependensi perlu ditentukan berdasarkan import pada kode, versi TensorFlow/TensorFlow Lite yang digunakan, dan lingkungan eksperimen. Versi dependensi sebaiknya dikunci untuk mendukung reproduksibilitas.

### 9.4 Siapkan Dataset

Unduh dataset dari [Mendeley Data](https://data.mendeley.com/datasets/zp4nf2dxbh/1), lalu letakkan berkas audio pada struktur direktori yang diharapkan oleh konfigurasi eksperimen:

```text
Data/
└── PoultryVocalization/
    └── Chicken_Audio_Dataset/
        ├── Healthy/
        ├── Noise/
        └── Unhealthy/
```

Nama folder dan lokasi berkas harus disesuaikan dengan konfigurasi aktual. Periksa `experiment/Baseline/config.py` sebelum menjalankan pipeline.

## 10. Menjalankan Eksperimen

Perintah berikut menunjukkan urutan eksekusi berdasarkan skrip yang tersedia. Jalankan dari direktori root repositori dan periksa konfigurasi serta dependensi sebelum memulai.

### 10.1 Audit Dataset dan Pembagian Data

```bash
python experiment/Baseline/audit_dataset.py
python experiment/Baseline/split_dataset.py
```

Pembagian data harus konsisten untuk seluruh eksperimen. Jika `split_manifest.csv` yang digunakan untuk menghasilkan hasil terdahulu masih tersedia, simpan salinannya sebelum membuat ulang split.

### 10.2 Melatih Model

```bash
python experiment/Baseline/train.py
python experiment/CompactCNN/train.py
python experiment/DSCNN/train.py
python experiment/TCResNet/train.py
```

Pastikan setiap konfigurasi mengacu pada dataset, manifest split, dan preprocessing yang sesuai. Seluruh model (termasuk Baseline) telah diselaraskan menggunakan modul preprocessing bersama (`src/pipeline.py`) dan arsitektur resmi pada direktori `models/`.

### 10.3 Kuantisasi dan Evaluasi INT8

```bash
python experiment/int8/quantize.py
python experiment/int8/evaluate.py
```

Jalankan tahap ini setelah model TC-ResNet8 yang sesuai tersedia. Periksa jalur model input dan output dalam konfigurasi sebelum melakukan konversi atau evaluasi.

### 10.4 Evaluasi dan Perbandingan

Modul evaluasi yang tersedia:

- `evaluation/metrics.py`
- `evaluation/efficiency.py`
- `evaluation/comparison.py`

Cara menjalankannya dapat berbeda berdasarkan implementasi argumen pada masing-masing skrip. Periksa bagian konfigurasi dan entry point sebelum mengeksekusi modul tersebut secara mandiri.

## 11. Artefak Hasil

Artefak metrik yang digunakan dalam perbandingan meliputi:

- `artifacts/baseline/test_metrics.json`
- `artifacts/compact_cnn/test_metrics.json`
- `artifacts/ds_cnn/test_metrics.json`
- `artifacts/tcresnet/test_metrics.json`
- `artifacts/int8/test_metrics_int8.json`

Artefak tersebut menjadi sumber angka evaluasi yang dicantumkan dalam README ini. Ketersediaan berkas pada repositori publik perlu diperiksa secara langsung; jika tidak disertakan, hasil harus diregenerasi melalui eksperimen atau disediakan secara terpisah.

## 12. Keterbatasan Penelitian

Beberapa keterbatasan yang perlu diperhatikan:

1. **Ukuran dataset dan test set terbatas.** Hasil evaluasi belum cukup untuk menyimpulkan performa pada seluruh populasi ayam petelur atau lingkungan peternakan.
2. **Pemisahan berdasarkan sumber belum terverifikasi.** Pembagian file-level tidak menjamin pemisahan berdasarkan identitas ayam, sesi, atau lokasi perekaman.
3. **Label bukan diagnosis klinis.** Kelas Unhealthy merupakan label dataset, bukan konfirmasi penyakit tertentu.
4. **Performa DS-CNN belum memadai.** Recall Noise sebesar 0% menunjukkan perlunya evaluasi terhadap konfigurasi, preprocessing, dan proses pelatihan.
5. **Konsistensi preprocessing telah diverifikasi.** Skrip Baseline telah diselaraskan dengan `src/pipeline.py`, dan hasil pelatihan ulang membuktikan metrik yang identik persis dengan evaluasi historis (Akurasi 96,15%, Macro-F1 95,55%, Unhealthy Recall 100,00%).
6. **Validasi INT8 telah dikonfirmasi.** Pemeriksaan metadata membuktikan model merupakan *pure full-integer INT8* (I/O bertipe int8). Analisis komparasi *file-by-file* pada 52 sampel test set membuktikan kesesuaian prediksi 100% dengan versi FP32, didukung oleh pergeseran probabilitas yang sangat minim (MAE = 0,0043).
7. **Belum ada validasi hardware fisik.** Meskipun analisis siklus hidup tensor membuktikan model INT8 secara teoritis fit di internal SRAM (~18 KB kebutuhan minimum, alokasi arena ~32 KB dari ~350 KB usable heap) dan berkas C-array `artifacts/int8/model_data.h` telah siap, pengujian fisik berupa pengukuran latensi nyata (ms), konsumsi daya (mA), dan profiling heap runtime pada board fisik ESP32-S3 masih merupakan tahapan lanjutan.

## 13. Rencana Pengembangan

Pengembangan selanjutnya dapat mencakup:

- Pengujian inferensi firmware pada modul fisik ESP32-S3 menggunakan TensorFlow Lite for Microcontrollers dan akselerasi ESP-NN.
- Evaluasi variasi parameter ekstraksi fitur (misalnya hop length dan window size MFCC).
- Menyimpan versi dependensi dan konfigurasi eksperimen.
- Memastikan dataset, split manifest, model terlatih, dan artefak evaluasi dapat direproduksi.
- Menambahkan pengujian pada data eksternal atau pembagian berbasis sesi/sumber apabila metadata tersedia.
- Mengembangkan prototipe pemantauan audio setelah kelayakan model tervalidasi di lapangan.

## 14. Kesimpulan

Eksperimen komparatif L-Avera-EFEST menunjukkan bahwa optimasi arsitektur neural network mampu menekan kebutuhan komputasi secara signifikan tanpa mengorbankan sensitivitas deteksi kritis:

1. **Baseline CNN** memperoleh akurasi tertinggi pada test set yang digunakan (96,15%, Macro-F1 95,55%), namun menuntut 886.979 parameter dan 9,18 juta MACCs (ukuran berkas ±10,2 MB).
2. **TC-ResNet8** menunjukkan trade-off paling unggul dengan mempertahankan **Recall Unhealthy 100,00%** (18/18 kasus terdeteksi) dan performa klasifikasi tinggi (Akurasi 94,23%, Macro-F1 93,24%), sembari memangkas **92,6% parameter** (hanya 65.723 parameter) serta menghemat **67,0% beban komputasi MACCs** (3,03 juta operasi).
3. **Kuantisasi Post-Training (INT8)** berhasil memadatkan model TC-ResNet8 menjadi **95,45 KB** (reduksi 89,73%) dengan kesesuaian prediksi 100% pada test set dan pergeseran probabilitas minimal (MAE = 0,0043).
4. **Kelayakan Edge:** Analisis memori menunjukkan kebutuhan Tensor Arena hanya **~18–32 KB**, memungkinkan model berjalan mandiri pada **Internal SRAM ESP32-S3** tanpa memerlukan keping eksternal PSRAM. Berkas header `artifacts/int8/model_data.h` telah tervalidasi dan siap digunakan untuk pengembangan firmware embedded.

## Referensi Dataset

Poultry Vocalization Signal Dataset for Early Disease Detection. Mendeley Data. https://doi.org/10.17632/zp4nf2dxbh.1

---

**Disclaimer:** L-Avera-EFEST merupakan proyek penelitian dan pengembangan model klasifikasi audio. Sistem ini bukan alat diagnosis veteriner dan hasil klasifikasinya tidak boleh digunakan sebagai satu-satunya dasar untuk menentukan status kesehatan ayam.
