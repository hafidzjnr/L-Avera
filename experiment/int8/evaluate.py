# Evaluasi model TFLite INT8 pada held-out test set
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from experiment.int8.config import (
    ARTIFACT_DIR,
    BASELINE_SPLIT_PATH,
    CANDIDATE_MODEL_KEY,
    CLASSES,
    MAX_FRAMES,
    METRICS_PATH,
    N_MFCC,
    RANDOM_SEED,
    SAMPLE_RATE,
    TFLITE_MODEL_PATH,
)
from src.pipeline import make_arrays


# --- Inferensi Kuantisasi INT8 per Sampel ---
def _run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    x: np.ndarray,
) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]["quantization"][0]
    input_zero = input_details[0]["quantization"][1]
    output_scale = output_details[0]["quantization"][0]
    output_zero = output_details[0]["quantization"][1]

    preds = []
    for sample in x:
        # Kuantisasi input float ke int8
        inp = sample[np.newaxis].astype(np.float32)
        inp_q = np.round(inp / input_scale + input_zero).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], inp_q)
        interpreter.invoke()

        # Dekuantisasi output int8 ke probabilitas float
        out_q = interpreter.get_tensor(output_details[0]["index"])
        out_f = (out_q.astype(np.float32) - output_zero) * output_scale
        preds.append(out_f[0])
    return np.array(preds)


def main() -> None:
    # Validasi ketersediaan model TFLite
    if not TFLITE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model TFLite tidak ditemukan: {TFLITE_MODEL_PATH}\n"
            "Jalankan: python -m experiment.int8.quantize"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Memuat split data pengujian (test set)
    split = pd.read_csv(BASELINE_SPLIT_PATH)
    test_df = split[split.split == "test"]
    if len(test_df) == 0:
        raise ValueError("Split test kosong.")

    # Ekstraksi fitur audio test set
    print("Mengekstrak fitur test (augmentasi: OFF)...")
    x_test, y_test = make_arrays(
        test_df, CLASSES, SAMPLE_RATE, N_MFCC, MAX_FRAMES,
        training=False, artifact_dir=ARTIFACT_DIR, random_seed=RANDOM_SEED,
    )

    # Inisialisasi TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()

    # Eksekusi inferensi INT8
    print(f"Menjalankan inferensi INT8 pada {len(x_test)} sampel test...")
    probabilities = _run_tflite_inference(interpreter, x_test)
    y_pred = np.argmax(probabilities, axis=1)

    # Perhitungan metrik evaluasi
    accuracy = float(accuracy_score(y_test, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    unhealthy_i = CLASSES.index("Unhealthy")
    report = classification_report(
        y_test, y_pred, labels=range(len(CLASSES)),
        target_names=CLASSES, zero_division=0, output_dict=True,
    )
    model_size_kb = TFLITE_MODEL_PATH.stat().st_size / 1024

    # Penyimpanan hasil metrik INT8 ke JSON
    metrics = {
        "model": f"{CANDIDATE_MODEL_KEY}_int8",
        "precision": "INT8",
        "model_size_kb": round(model_size_kb, 2),
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "unhealthy_precision": float(p[unhealthy_i]),
        "unhealthy_recall": float(r[unhealthy_i]),
        "unhealthy_f1": float(f[unhealthy_i]),
        "per_class_report": report,
        "class_order": list(CLASSES),
        "test_original_file_count": int(len(test_df)),
        "test_feature_count": int(len(y_test)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nHASIL TEST AKHIR — INT8")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "per_class_report"},
        indent=2,
    ))
    print("\nLaporan Klasifikasi:")
    print(classification_report(
        y_test, y_pred, labels=range(len(CLASSES)),
        target_names=CLASSES, zero_division=0, digits=4,
    ))

    # Visualisasi Confusion Matrix INT8
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"{CANDIDATE_MODEL_KEY} INT8 — Held-out Test Set")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\nMetrik tersimpan: {METRICS_PATH}")


if __name__ == "__main__":
    main()
