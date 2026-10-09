# Kuantisasi INT8 Post-Training (PTQ) via TensorFlow Lite
# Mengonversi model Keras terlatih menjadi integer penuh menggunakan dataset representatif dari data training
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from experiment.int8.config import (
    ARTIFACT_DIR,
    BASELINE_SPLIT_PATH,
    CANDIDATE_MODEL_PATH,
    CLASSES,
    MAX_FRAMES,
    N_CALIBRATION_SAMPLES,
    N_MFCC,
    RANDOM_SEED,
    SAMPLE_RATE,
    TFLITE_MODEL_PATH,
)
from src.pipeline import make_arrays


# --- Generator Kalibrasi untuk PTQ ---
def _representative_dataset(x_calib: np.ndarray):
    for sample in x_calib:
        yield [sample[np.newaxis].astype(np.float32)]


def main() -> None:
    # Validasi path model kandidat
    if not CANDIDATE_MODEL_PATH or not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Tentukan CANDIDATE_MODEL_PATH pada experiment/int8/config.py "
            "sebelum menjalankan kuantisasi."
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Memuat subset sampel training untuk kalibrasi (menghindari data leakage dari test set)
    split = pd.read_csv(BASELINE_SPLIT_PATH)
    train_df = split[split.split == "train"]
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(
        len(train_df),
        size=min(N_CALIBRATION_SAMPLES, len(train_df)),
        replace=False,
    )
    calib_df = train_df.iloc[sample_idx]

    # Ekstraksi fitur audio untuk kalibrasi
    print(f"Mengekstrak {len(calib_df)} sampel kalibrasi (tanpa augmentasi)...")
    x_calib, _ = make_arrays(
        calib_df, CLASSES, SAMPLE_RATE, N_MFCC, MAX_FRAMES,
        training=False, artifact_dir=ARTIFACT_DIR, random_seed=RANDOM_SEED,
    )

    # Konversi model ke format INT8 TFLite
    print(f"Memuat model kandidat: {CANDIDATE_MODEL_PATH}")
    model = tf.keras.models.load_model(str(CANDIDATE_MODEL_PATH))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(x_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Mengonversi model ke format INT8 TFLite...")
    tflite_model = converter.convert()
    TFLITE_MODEL_PATH.write_bytes(tflite_model)

    # Informasi ukuran berkas model hasil kuantisasi
    size_kb = TFLITE_MODEL_PATH.stat().st_size / 1024
    print(f"\nModel INT8 berhasil disimpan: {TFLITE_MODEL_PATH}")
    print(f"Ukuran model: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
