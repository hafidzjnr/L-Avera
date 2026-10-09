# Model B — Compact 1D-CNN (Filter: 32 -> 64 -> 128)
# Reduksi separuh jumlah filter dari baseline untuk komputasi lebih ringan
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D,
)
from tensorflow.keras.models import Sequential


def build_model(
    max_frames: int = 200,
    n_mfcc: int = 40,
    n_classes: int = 3,
) -> tf.keras.Model:
    model = Sequential(
        [
            # Lapisan Input
            tf.keras.layers.Input(shape=(max_frames, n_mfcc)),

            # Blok Konvolusi 1 (32 Filter)
            Conv1D(32, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            # Blok Konvolusi 2 (64 Filter)
            Conv1D(64, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            # Blok Konvolusi 3 (128 Filter)
            Conv1D(128, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.4),

            # Lapisan Klasifikasi (Fully Connected)
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(n_classes, activation="softmax"),
        ],
        name="compact_cnn",
    )

    # Kompilasi Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
