# Konfigurasi eksperimen kuantisasi INT8 (Post-Training Quantization)
# Model kandidat dipilih setelah perbandingan metrik FP32 selesai dievaluasi
from pathlib import Path

# --- Parameter Audio (Konsisten dengan Baseline) ---
CLASSES = ("Healthy", "Noise", "Unhealthy")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 200
RANDOM_SEED = 42

# --- Model Kandidat (Diisi setelah evaluasi perbandingan model) ---
# Contoh: CANDIDATE_MODEL_KEY = "compact_cnn"
#         CANDIDATE_MODEL_PATH = Path("artifacts/compact_cnn/CompactCNN.keras")
CANDIDATE_MODEL_KEY = "tcresnet"
CANDIDATE_MODEL_PATH = Path("artifacts/tcresnet/TCResNet8.keras")

# --- Data Kalibrasi (Menggunakan subset split training, tanpa test set) ---
BASELINE_SPLIT_PATH = Path("artifacts/baseline/split_manifest.csv")
N_CALIBRATION_SAMPLES = 100

# --- Jalur Penyimpanan Output Model INT8 ---
ARTIFACT_DIR = Path("artifacts/int8")
TFLITE_MODEL_PATH = ARTIFACT_DIR / "model_int8.tflite"
METRICS_PATH = ARTIFACT_DIR / "test_metrics_int8.json"
