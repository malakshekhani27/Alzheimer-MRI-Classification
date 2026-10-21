import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = ["CN", "MCI", "AD"]


def collect_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_items(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw dataset directory")
    parser.add_argument("--output", required=True, help="Output split directory")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise ValueError("Train + validation ratios must be > 0 and < 1.")

    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Input folder not found: {source}")

    rng = random.Random(args.seed)

    class_dirs = [source / c for c in DEFAULT_CLASSES]
    missing = [str(p) for p in class_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing class folders: " + ", ".join(missing) +
            ". Expected CN, MCI and AD."
        )

    if output.exists():
        shutil.rmtree(output)

    for split in ("train", "val", "test"):
        for cls in DEFAULT_CLASSES:
            (output / split / cls).mkdir(parents=True, exist_ok=True)

    print("Preparing dataset...")
    for cls in DEFAULT_CLASSES:
        items = collect_images(source / cls)
        if len(items) < 3:
            raise ValueError(f"Class {cls} has fewer than 3 images.")

        rng.shuffle(items)
        train_items, val_items, test_items = split_items(
            items, args.train, args.val
        )

        for split_name, split_items_list in (
            ("train", train_items),
            ("val", val_items),
            ("test", test_items),
        ):
            for index, src in enumerate(split_items_list):
                # Unique destination name avoids collisions from nested folders.
                dst = output / split_name / cls / f"{index:06d}_{src.name}"
                shutil.copy2(src, dst)

        print(
            f"{cls}: total={len(items)}, "
            f"train={len(train_items)}, val={len(val_items)}, test={len(test_items)}"
        )

    print(f"Done. Dataset written to: {output}")


if __name__ == "__main__":
    main()
