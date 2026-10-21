import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


NUM_CLASSES = 4

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]


def build_model(input_shape=(224, 224, 3)):
    """
    Build EfficientNetB0 model for
    4-class Alzheimer's MRI classification.
    """

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Initially freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model


def unfreeze_for_finetuning(model):
    """
    Unfreeze the last part of EfficientNet
    for fine-tuning.
    """

    # Find EfficientNet base model
    base_model = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):
            base_model = layer
            break

    if base_model is not None:

        base_model.trainable = True

        # Freeze earlier layers
        for layer in base_model.layers[:-30]:

            layer.trainable = False

        # Keep BatchNorm frozen
        for layer in base_model.layers:

            if isinstance(
                layer,
                layers.BatchNormalization
            ):
                layer.trainable = False

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()

    print("\nModel created successfully!")

    print(
        "Number of classes:",
        NUM_CLASSES
    )