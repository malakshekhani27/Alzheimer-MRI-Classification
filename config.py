from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


# --------------------------------------------------
# Model and output paths
# --------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "alzheimer_efficientnet.keras"

CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"


# --------------------------------------------------
# Image / training settings
# --------------------------------------------------

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

SEED = 42


# --------------------------------------------------
# Alzheimer's MRI classes
# --------------------------------------------------

NUM_CLASSES = 4

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

for directory in (
    MODEL_DIR,
    OUTPUT_DIR
):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )