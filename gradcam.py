import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

from config import IMAGE_SIZE


def find_last_conv_layer(model):
    """
    Finds the last convolutional layer inside the EfficientNet backbone.
    """

    # Check nested models first
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):

            for sub_layer in reversed(layer.layers):
                try:
                    output_shape = sub_layer.output.shape

                    if len(output_shape) == 4:
                        return layer, sub_layer

                except Exception:
                    pass

    # Fallback: search top-level layers
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape

            if len(output_shape) == 4:
                return model, layer

        except Exception:
            pass

    raise ValueError("Could not find a suitable convolutional layer.")


def make_gradcam_heatmap(
    img_array,
    model,
    class_index=None,
    layer_name=None
):
    """
    Generate Grad-CAM heatmap for the selected class.
    """

    # Find EfficientNet backbone and last convolutional layer
    backbone, conv_layer = find_last_conv_layer(model)

    print("Grad-CAM backbone:", backbone.name)
    print("Grad-CAM layer:", conv_layer.name)

    # Create model that returns:
    # 1. convolutional feature maps
    # 2. backbone output
    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            conv_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        # Forward pass through EfficientNet
        conv_outputs, backbone_output = grad_model(
            img_array,
            training=False
        )

        # Apply the same classification head as the original model
        x = backbone_output

        # Global average pooling
        x = model.get_layer(
            "global_average_pooling2d"
        )(x)

        # Dropout layer
        x = model.get_layer(
            "dropout"
        )(x, training=False)

        # Final classifier
        predictions = model.get_layer(
            "dense"
        )(x)

        # Select class automatically
        if class_index is None:
            class_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[:, class_index]

    # Calculate gradients
    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    if grads is None:
        raise ValueError(
            "Gradients could not be calculated for Grad-CAM."
        )

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weight feature maps by gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


def overlay_heatmap(
    original_image,
    heatmap,
    alpha=0.40
):
    """
    Overlay Grad-CAM heatmap on original MRI image.
    """

    original = (
        original_image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    original_array = (
        np.asarray(original)
        .astype(np.float32)
        / 255.0
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    cmap = cm.get_cmap("jet")

    colored = cmap(
        heatmap_uint8
    )[:, :, :3]

    colored = np.asarray(
        Image.fromarray(
            np.uint8(colored * 255)
        ).resize(IMAGE_SIZE)
    ).astype(np.float32) / 255.0

    overlay = (
        (1 - alpha) * original_array
        + alpha * colored
    )

    return np.clip(
        overlay,
        0,
        1
    )