# Model D — TC-ResNet8 (Temporal Convolutional ResNet)
# Arsitektur konvolusi temporal 1D dengan residual shortcuts (referensi: Choi et al., 2019)
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


# --- Blok Residual TC-ResNet ---
def _residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 9,
    stride: int = 1,
) -> tf.Tensor:
    residual = x

    # Jalur konvolusi utama
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Proyeksi shortcut 1x1 jika dimensi channel berubah atau terdapat stride > 1
    if residual.shape[-1] != filters or stride != 1:
        residual = layers.Conv1D(filters, 1, strides=stride, padding="same")(
            residual
        )
        residual = layers.BatchNormalization()(residual)

    # Penjumlahan residual shortcut dan aktivasi
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    return x


# --- Pembangunan Arsitektur TC-ResNet8 ---
def build_model(
    max_frames: int = 200,
    n_mfcc: int = 40,
    n_classes: int = 3,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_frames, n_mfcc))

    # Lapisan Stem Konvolusi
    x = layers.Conv1D(16, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Tumpukan Blok Residual (ResBlock 1 - 3)
    x = _residual_block(x, filters=24, kernel_size=9, stride=2)
    x = layers.Dropout(0.3)(x)

    x = _residual_block(x, filters=32, kernel_size=9, stride=2)
    x = layers.Dropout(0.3)(x)

    x = _residual_block(x, filters=48, kernel_size=9, stride=2)
    x = layers.Dropout(0.4)(x)

    # Pooling Global dan Klasifikasi Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="tc_resnet8")

    # Kompilasi Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
