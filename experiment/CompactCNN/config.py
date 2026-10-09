# Konfigurasi eksperimen Compact CNN (Model B)
# Seluruh parameter preprocessing disamakan persis dengan baseline untuk perbandingan yang valid
from pathlib import Path

# --- Dataset Audio ---
DATASET_DIR = Path("Data") / "PoultryVocalization" / "Chicken_Audio_Dataset"
CLASSES = ("Healthy", "Noise", "Unhealthy")

# --- Ekstraksi Fitur MFCC (Sesuai Baseline) ---
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 200

# --- Reproduksibilitas ---
RANDOM_SEED = 42

# --- Parameter Pelatihan ---
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3

# --- Jalur File Split & Artifact Model ---
BASELINE_SPLIT_PATH = Path("artifacts/baseline/split_manifest.csv")
ARTIFACT_DIR = Path("artifacts/compact_cnn")
MODEL_PATH = ARTIFACT_DIR / "CompactCNN.keras"
