import os
import glob
import shutil
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_DIR = "data/hf_dataset/data"
OUTPUT_DIR = "data/processed"

LABEL_NAMES = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented",
}


# -------------------------------------------------
# Create folders
# -------------------------------------------------

for split in ["train", "val", "test"]:
    for class_name in LABEL_NAMES.values():
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )


# -------------------------------------------------
# Find parquet files
# -------------------------------------------------

train_files = glob.glob(
    os.path.join(DATASET_DIR, "train*.parquet")
)

test_files = glob.glob(
    os.path.join(DATASET_DIR, "test*.parquet")
)

if not train_files:
    raise FileNotFoundError("Training parquet file not found.")

if not test_files:
    raise FileNotFoundError("Test parquet file not found.")


print("Training file:")
print(train_files[0])

print("\nTest file:")
print(test_files[0])


# -------------------------------------------------
# Load datasets
# -------------------------------------------------

train_df = pd.read_parquet(train_files[0])
test_df = pd.read_parquet(test_files[0])

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))


# -------------------------------------------------
# Convert image bytes to PIL image
# -------------------------------------------------

def save_image(image_data, output_path):

    try:

        # Hugging Face image field is usually
        # stored as {'bytes': b'...'}

        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes")

            if image_bytes is not None:
                image = Image.open(
                    BytesIO(image_bytes)
                )
            else:
                raise ValueError(
                    "Image bytes not found."
                )

        elif isinstance(image_data, bytes):

            image = Image.open(
                BytesIO(image_data)
            )

        else:

            raise ValueError(
                f"Unsupported image type: {type(image_data)}"
            )

        # Convert to RGB
        image = image.convert("RGB")

        # Resize
        image = image.resize((224, 224))

        # Save
        image.save(output_path, "JPEG", quality=95)

    except Exception as e:

        print(
            f"Could not save image: {e}"
        )


# -------------------------------------------------
# Save test dataset
# -------------------------------------------------

print("\nConverting TEST dataset...")

for index, row in test_df.iterrows():

    label = int(row["label"])

    class_name = LABEL_NAMES[label]

    output_path = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name,
        f"test_{index:05d}.jpg"
    )

    save_image(
        row["image"],
        output_path
    )

    if (index + 1) % 100 == 0:
        print(
            f"Test images processed: {index + 1}"
        )


# -------------------------------------------------
# Split original training dataset
# into TRAIN and VALIDATION
# -------------------------------------------------

print("\nCreating train/validation split...")

train_part, val_part = train_test_split(
    train_df,
    test_size=0.15,
    random_state=42,
    stratify=train_df["label"]
)


# -------------------------------------------------
# Save TRAIN images
# -------------------------------------------------

print("\nConverting TRAIN dataset...")

for index, row in train_part.iterrows():

    label = int(row["label"])

    class_name = LABEL_NAMES[label]

    output_path = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name,
        f"train_{index:05d}.jpg"
    )

    save_image(
        row["image"],
        output_path
    )


# -------------------------------------------------
# Save VALIDATION images
# -------------------------------------------------

print("\nConverting VALIDATION dataset...")

for index, row in val_part.iterrows():

    label = int(row["label"])

    class_name = LABEL_NAMES[label]

    output_path = os.path.join(
        OUTPUT_DIR,
        "val",
        class_name,
        f"val_{index:05d}.jpg"
    )

    save_image(
        row["image"],
        output_path
    )


# -------------------------------------------------
# Print summary
# -------------------------------------------------

print("\n" + "=" * 50)
print("DATASET CONVERSION COMPLETE")
print("=" * 50)

print("\nTrain:", len(train_part))
print("Validation:", len(val_part))
print("Test:", len(test_df))

print("\nClasses:")

for label, name in LABEL_NAMES.items():

    train_count = sum(
        train_part["label"] == label
    )

    val_count = sum(
        val_part["label"] == label
    )

    test_count = sum(
        test_df["label"] == label
    )

    print(
        f"{name}: "
        f"train={train_count}, "
        f"val={val_count}, "
        f"test={test_count}"
    )

print("\nDataset saved to:")
print(OUTPUT_DIR)
