import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from config import MODEL_PATH, IMAGE_SIZE


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    with open("models/class_names.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]

    print("\nPrediction:")
    print(f"Class: {class_names[order[0]]}")
    print(f"Confidence: {probabilities[order[0]] * 100:.2f}%")

    print("\nAll probabilities:")
    for index in order:
        print(f"{class_names[index]}: {probabilities[index] * 100:.2f}%")

    return class_names[order[0]], probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first with: python train.py")

    predict_image(args.image)
