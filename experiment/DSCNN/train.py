# Pelatihan Model C (Depthwise Separable CNN / DS-CNN) dan evaluasi pada test set
# Menggunakan manifest split baseline yang sama agar perbandingan objektif
from __future__ import annotations

import json
import random

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from experiment.DSCNN.config import (
    ARTIFACT_DIR,
    BASELINE_SPLIT_PATH,
    BATCH_SIZE,
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LR_PATIENCE,
    MAX_FRAMES,
    MODEL_PATH,
    N_MFCC,
    RANDOM_SEED,
    SAMPLE_RATE,
)
from models.ds_cnn import build_model
from src.pipeline import make_arrays


def main() -> None:
    # Inisialisasi seed dan direktori output
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Validasi keberadaan file split manifest baseline
    if not BASELINE_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline split manifest not found: {BASELINE_SPLIT_PATH}\n"
            "Jalankan split baseline terlebih dahulu: "
            "python -m experiment.Baseline.split_dataset"
        )

    split = pd.read_csv(BASELINE_SPLIT_PATH)
    train_df = split[split.split == "train"]
    val_df = split[split.split == "validation"]
    test_df = split[split.split == "test"]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Terdapat split yang kosong pada manifest.")

    # Ekstraksi fitur audio MFCC
    print("Mengekstrak fitur training (augmentasi: ON)...")
    x_train, y_train = make_arrays(
        train_df, CLASSES, SAMPLE_RATE, N_MFCC, MAX_FRAMES,
        training=True, artifact_dir=ARTIFACT_DIR, random_seed=RANDOM_SEED,
    )
    print("Mengekstrak fitur validation (augmentasi: OFF)...")
    x_val, y_val = make_arrays(
        val_df, CLASSES, SAMPLE_RATE, N_MFCC, MAX_FRAMES,
        training=False, artifact_dir=ARTIFACT_DIR, random_seed=RANDOM_SEED,
    )
    print("Mengekstrak fitur test (augmentasi: OFF)...")
    x_test, y_test = make_arrays(
        test_df, CLASSES, SAMPLE_RATE, N_MFCC, MAX_FRAMES,
        training=False, artifact_dir=ARTIFACT_DIR, random_seed=RANDOM_SEED,
    )
    print(f"Bentuk data -> train: {x_train.shape}  val: {x_val.shape}  test: {x_test.shape}")

    # Membangun arsitektur model DS-CNN
    model = build_model(
        max_frames=MAX_FRAMES, n_mfcc=N_MFCC, n_classes=len(CLASSES)
    )
    model.summary()

    # Pengaturan callback pelatihan (penurunan learning rate & early stopping)
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", patience=LR_PATIENCE, factor=0.5,
            min_lr=1e-5, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
    ]

    # Pelatihan model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
    )

    # Prediksi dan evaluasi pada held-out test set
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
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

    # Penyimpanan metrik hasil pengujian
    metrics = {
        "model": "ds_cnn",
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "unhealthy_precision": float(p[unhealthy_i]),
        "unhealthy_recall": float(r[unhealthy_i]),
        "unhealthy_f1": float(f[unhealthy_i]),
        "per_class_report": report,
        "class_order": list(CLASSES),
        "random_seed": RANDOM_SEED,
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "max_frames": MAX_FRAMES,
        "test_original_file_count": int(len(test_df)),
        "test_feature_count": int(len(y_test)),
    }
    (ARTIFACT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nHASIL TEST AKHIR — DS-CNN")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "per_class_report"},
        indent=2,
    ))
    print("\nLaporan Klasifikasi:")
    print(classification_report(
        y_test, y_pred, labels=range(len(CLASSES)),
        target_names=CLASSES, zero_division=0, digits=4,
    ))

    # Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title("DS-CNN — Held-out Test Set")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Visualisasi grafik histori akurasi dan loss
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history.get(metric, []), label=f"train_{metric}")
        ax.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"Grafik Pelatihan — {metric}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / f"{metric}_history.png", dpi=200)
        plt.close(fig)

    # Simpan model final
    model.save(MODEL_PATH)
    print(f"\nModel tersimpan : {MODEL_PATH}")
    print(f"Metrik tersimpan: {ARTIFACT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
