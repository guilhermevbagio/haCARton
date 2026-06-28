import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["aprovar", "rejeitar", "revisar"]
IMG_SIZE = (224, 224)
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def ensure_rgb_pil(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def load_image_pil(image_path: str | Path) -> Image.Image:
    return ensure_rgb_pil(Image.open(image_path))


def prepare_image_for_model(image: Image.Image | np.ndarray, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image

    pil_image = ensure_rgb_pil(pil_image).resize(img_size)
    array = np.asarray(pil_image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def save_classes_json(class_mapping: Dict[int, str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {str(idx): name for idx, name in class_mapping.items()}
    output_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")


def load_classes_json(input_path: str | Path) -> Dict[int, str]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return {int(idx): name for idx, name in data.items()}


def count_images_by_class(dataset_dir: str | Path, classes: Iterable[str] = CLASS_NAMES) -> Dict[str, int]:
    dataset_dir = Path(dataset_dir)
    counts: Dict[str, int] = {}
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(1 for path in class_dir.iterdir() if path.is_file())
    return counts


def _collect_class_files(dataset_dir: Path, class_name: str) -> List[Path]:
    class_dir = dataset_dir / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.is_file())


def split_dataset_if_needed(
    source_dir: str | Path = "dataset_rotulado",
    output_dir: str | Path = "dataset_validador",
    classes: Iterable[str] = CLASS_NAMES,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> Path:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    train_root = output_dir / "train"
    val_root = output_dir / "val"

    if train_root.exists() and val_root.exists():
        return output_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset de origem n?o encontrado: {source_dir}")

    random.seed(seed)
    for class_name in classes:
        files = _collect_class_files(source_dir, class_name)
        if not files:
            continue

        shuffled = files[:]
        random.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * validation_split)) if len(shuffled) > 1 else 0
        val_files = shuffled[:val_count]
        train_files = shuffled[val_count:]
        if not train_files and val_files:
            train_files = [val_files.pop()]

        for split_name, split_files in (("train", train_files), ("val", val_files)):
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for file_path in split_files:
                destination = split_dir / file_path.name
                if not destination.exists():
                    shutil.copy2(file_path, destination)

    return output_dir


def build_datasets_from_directory(
    dataset_dir: str | Path,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = 16,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    return train_ds, val_ds, class_names


def summarize_split_counts(dataset_dir: str | Path, classes: Iterable[str] = CLASS_NAMES) -> Dict[str, Dict[str, int]]:
    dataset_dir = Path(dataset_dir)
    return {
        "train": count_images_by_class(dataset_dir / "train", classes),
        "val": count_images_by_class(dataset_dir / "val", classes),
    }
