import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import TEST_DIR, MODEL_PATH, OUTPUT_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        label_mode="categorical",
    )

    class_names = ds.class_names
    model = tf.keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in ds],
        axis=0,
    )

    print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
