
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    Rescaling,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model


def build_model(image_size=160, trainable_base=False):

    # Input layer
    inputs = Input(
        shape=(image_size, image_size, 3)
    )

    # Convert pixel values from [0, 1] to [-1, 1]
    x = Rescaling(
        scale=2.0,
        offset=-1.0
    )(inputs)

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = trainable_base

    # Extract image features
    x = base_model(
        x,
        training=False
    )

    # Classification layers
    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.4)(x)

    # Binary classification
    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    return model
