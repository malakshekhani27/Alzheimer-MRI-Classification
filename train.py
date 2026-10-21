import os
import tensorflow as tf

from model import build_model, unfreeze_for_finetuning
from data_loader import load_datasets


EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 5

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "alzheimer_efficientnet.keras"
)


def main():

    # --------------------------------------------------
    # Create model directory
    # --------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("Classes:", class_names)

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = build_model()

    model.summary()

    # ==================================================
    # PHASE 1
    # Train classifier with EfficientNet frozen
    # ==================================================

    print("\nPhase 1: training classifier...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1
        )
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks
    )

    # ==================================================
    # PHASE 2
    # Fine-tuning
    # ==================================================

    print("\nPhase 2: fine-tuning EfficientNet...")

    model = unfreeze_for_finetuning(model)

    # IMPORTANT:
    # Recompile after changing trainable layers.

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks
    )

    # --------------------------------------------------
    # Save final model
    # --------------------------------------------------

    model.save(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nModel saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------

    print("\nEvaluating on test dataset...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()