# Analisis efisiensi model: jumlah parameter komputasi dan ukuran berkas
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

# Jalur model format Keras FP32
MODEL_PATHS: dict[str, Path] = {
    "baseline_cnn": Path("artifacts/baseline/Baseline.keras"),
    "compact_cnn":  Path("artifacts/compact_cnn/CompactCNN.keras"),
    "ds_cnn":       Path("artifacts/ds_cnn/DSCNN.keras"),
    "tc_resnet8":   Path("artifacts/tcresnet/TCResNet8.keras"),
}

# Jalur model format TFLite INT8
INT8_MODEL_PATH = Path("artifacts/int8/model_int8.tflite")


# --- Penghitungan Parameter Model ---
def count_parameters(model_path: Path) -> dict[str, int]:
    model = tf.keras.models.load_model(str(model_path))
    trainable = int(sum(
        tf.size(w).numpy() for w in model.trainable_weights
    ))
    non_trainable = int(sum(
        tf.size(w).numpy() for w in model.non_trainable_weights
    ))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


# --- Penghitungan Ukuran Berkas ---
def model_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 2)


# --- Pembuatan Laporan Efisiensi Per-Model ---
def efficiency_report(model_key: str) -> dict:
    path = MODEL_PATHS[model_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas model tidak ditemukan untuk '{model_key}': {path}"
        )
    return {
        "model": model_key,
        "parameters": count_parameters(path),
        "size_kb": model_size_kb(path),
    }


# --- Pembuatan Laporan Efisiensi Seluruh Model ---
def all_efficiency_reports() -> list[dict]:
    reports = []
    for key, path in MODEL_PATHS.items():
        if path.exists():
            reports.append(efficiency_report(key))
    if INT8_MODEL_PATH.exists():
        reports.append({
            "model": "int8",
            "parameters": None,
            "size_kb": model_size_kb(INT8_MODEL_PATH),
        })
    return reports
