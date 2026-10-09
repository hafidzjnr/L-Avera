"""Train the reference 1D-CNN baseline and evaluate the held-out test set."""
from __future__ import annotations

import json
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D,
)
from tensorflow.keras.models import Sequential

from experiment.Baseline.config import (
    ARTIFACT_DIR, BATCH_SIZE, CLASSES, EPOCHS, EARLY_STOPPING_PATIENCE,
    LR_PATIENCE, MAX_FRAMES, MODEL_PATH, N_MFCC, RANDOM_SEED,
    SAMPLE_RATE, SPLIT_PATH,
)


def extract_mfcc(path: str, augment: bool = False, rng=None) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio: {path}")

    if augment:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            if rng is None:
                rng = np.random.default_rng(RANDOM_SEED)
            scale = 0.005 * peak * float(rng.uniform(0.0, 1.0))
            audio = np.clip(audio + scale * rng.normal(size=audio.shape), -1.0, 1.0)

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.T.astype(np.float32)


def make_arrays(frame: pd.DataFrame, training: bool):
    rng = np.random.default_rng(RANDOM_SEED)
    xs, ys = [], []
    errors = []
    for row in frame.itertuples(index=False):
        try:
            xs.append(extract_mfcc(row.path, augment=False))
            ys.append(CLASSES.index(row.label))
            # Generate augmentation only after original files have been split,
            # and only for training records.
            if training:
                xs.append(extract_mfcc(row.path, augment=True, rng=rng))
                ys.append(CLASSES.index(row.label))
        except Exception as exc:
            errors.append({"path": row.path, "label": row.label, "error": str(exc)})
            print(f"[WARN] Skipping {row.path}: {exc}")

    if errors:
        pd.DataFrame(errors).to_csv(
            ARTIFACT_DIR / "feature_extraction_errors.csv", index=False
        )
    if not xs:
        raise RuntimeError("No audio features extracted for this split.")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def build_model():
    # Keep the original baseline's three convolution widths: 64, 128, 256.
    model = Sequential([
        tf.keras.layers.Input(shape=(MAX_FRAMES, N_MFCC)),
        Conv1D(64, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(256, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError("Run first: python -m Baseline.split_dataset")

    split = pd.read_csv(SPLIT_PATH)
    train_df = split[split.split == "train"]
    val_df = split[split.split == "validation"]
    test_df = split[split.split == "test"]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    print("Extracting training features; augmentation is train-only...")
    x_train, y_train = make_arrays(train_df, training=True)
    print("Extracting validation features without augmentation...")
    x_val, y_val = make_arrays(val_df, training=False)
    print("Extracting test features without augmentation...")
    x_test, y_test = make_arrays(test_df, training=False)

    print(f"train: {x_train.shape}; validation: {x_val.shape}; test: {x_test.shape}")

    model = build_model()
    model.summary()
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
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
    )

    # Test set is used only after training/model selection.
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
    metrics = {
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
    print("\nFINAL TEST RESULTS")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_report"},
                     indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=range(len(CLASSES)),
        target_names=CLASSES, zero_division=0, digits=4,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title("Baseline — Held-out Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history.get(metric, []), label=f"train_{metric}")
        ax.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"Training history — {metric}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / f"{metric}_history.png", dpi=200)
        plt.close(fig)

    model.save(MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {ARTIFACT_DIR / 'test_metrics.json'}")
    print(
        "\nResearch limitation: the default split is file-level. If metadata "
        "identifies chicken/session/source recording, perform a group-based split "
        "before making source-independent generalization claims."
    )


if __name__ == "__main__":
    main()
