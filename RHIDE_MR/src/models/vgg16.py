# src/models/vgg16.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_model(num_classes: int):
    """
    Full VGG16 model used by:
    - device
    - edge
    - cloud
    """
    base = VGG16(
        include_top=False,
        weights=None,          # IMPORTANT: no ImageNet leakage
        input_shape=(224, 224, 3),
    )

    x = base.output
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dense(4096, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

