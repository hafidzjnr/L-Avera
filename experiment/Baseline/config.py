"""Single source of truth for the ChickVoice baseline experiment."""
from pathlib import Path

DATASET_DIR = Path("Data") / "PoultryVocalization" / "Chicken_Audio_Dataset"

CLASSES = ("Healthy", "Noise", "Unhealthy")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 200

RANDOM_SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15  # fraction of the complete dataset
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3

ARTIFACT_DIR = Path("artifacts/baseline")
MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.csv"
SPLIT_PATH = ARTIFACT_DIR / "split_manifest.csv"
MODEL_PATH = ARTIFACT_DIR / "Baseline.keras"
