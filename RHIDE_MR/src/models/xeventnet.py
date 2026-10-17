# src/models/xeventnet.py
from __future__ import annotations

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense,
    Conv3D, MaxPooling3D, GlobalAveragePooling3D, Concatenate, Dropout
)
from tensorflow.keras import Model


def build_model(
    input_shape_2d: Tuple[int, int, int],
    input_shape_3d: Tuple[int, int, int, int],
    num_classes: int = 5,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    loss: str | tf.keras.losses.Loss = "sparse_categorical_crossentropy",
    metrics: Optional[list] = None,
) -> Model:
    """
    XevenetNet: two-branch 2D+3D architecture that merges for classification.
    Layer names are stable & human-readable to support split points.
    """
    if metrics is None:
        metrics = ["accuracy"]

    # 2D branch
    input_2d = Input(shape=input_shape_2d, name="input_2d")
    x2d = Conv2D(32, (3, 3), activation="relu", padding="same", name="2d_conv1")(input_2d)
    x2d = Conv2D(64, (3, 3), activation="relu", padding="same", name="2d_conv2")(x2d)
    x2d = MaxPooling2D((2, 2), name="2d_pool1")(x2d)
    x2d = Conv2D(128, (3, 3), activation="relu", padding="same", name="2d_conv3")(x2d)
    x2d = GlobalMaxPooling2D(name="2d_global_pool")(x2d)
    x2d = Dense(128, activation="relu", name="2d_dense")(x2d)

    # 3D branch
    input_3d = Input(shape=input_shape_3d, name="input_3d")
    x3d = Conv3D(32, (3, 3, 3), activation="relu", padding="same", name="3d_conv1")(input_3d)
    x3d = Conv3D(64, (3, 3, 3), activation="relu", padding="same", name="3d_conv2")(x3d)
    x3d = MaxPooling3D((2, 2, 2), name="3d_pool1")(x3d)
    x3d = Conv3D(128, (3, 3, 3), activation="relu", padding="same", name="3d_conv3")(x3d)
    x3d = GlobalAveragePooling3D(name="3d_global_pool")(x3d)
    x3d = Dense(128, activation="relu", name="3d_dense")(x3d)

    # Merge & head
    combined = Concatenate(name="concat")([x2d, x3d])
    combined = Dense(128, activation="relu", name="combined_dense")(combined)
    combined = Dropout(0.5, name="combined_dropout")(combined)
    output = Dense(num_classes, activation="softmax", name="output")(combined)

    model = Model(inputs=[input_2d, input_3d], outputs=output, name="XevenetNet")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

