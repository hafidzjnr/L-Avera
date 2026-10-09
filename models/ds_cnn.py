# Model C — Depthwise Separable 1D-CNN (DS-CNN)
# Menggantikan konvolusi standar dengan SeparableConv1D untuk efisiensi FLOPs dan parameter
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout, Flatten,
    MaxPooling1D, SeparableConv1D,
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

            # Blok Separable Conv 1 (32 Filter)
            SeparableConv1D(32, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            # Blok Separable Conv 2 (64 Filter)
            SeparableConv1D(64, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            # Blok Separable Conv 3 (128 Filter)
            SeparableConv1D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.4),

            # Lapisan Klasifikasi (Fully Connected)
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(n_classes, activation="softmax"),
        ],
        name="ds_cnn",
    )

    # Kompilasi Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
