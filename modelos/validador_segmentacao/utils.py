import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
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


def pil_or_array_to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image.astype(np.uint8)
        if array.ndim == 2:
            return np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 4:
            return np.asarray(Image.fromarray(array, mode="RGBA").convert("RGB"), dtype=np.uint8)
        return array[:, :, :3]
    return np.asarray(ensure_rgb_pil(image), dtype=np.uint8)


def prepare_image_for_model(image: Image.Image | np.ndarray, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image

    pil_image = ensure_rgb_pil(pil_image).resize(img_size)
    array = np.asarray(pil_image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def split_side_by_side_image(image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = pil_or_array_to_rgb_array(image)
    _height, width = rgb.shape[:2]
    mid = max(1, width // 2)
    left = rgb[:, :mid]
    right = rgb[:, mid:]
    if right.size == 0:
        right = left.copy()
    return left, right


def estimate_non_satellite_score(image: Image.Image | np.ndarray) -> Dict[str, float | bool]:
    original_half, _ = split_side_by_side_image(image)
    resized = cv2.resize(original_half, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    border = np.concatenate(
        [
            gray[:24, :].ravel(),
            gray[-24:, :].ravel(),
            gray[:, :24].ravel(),
            gray[:, -24:].ravel(),
        ]
    )
    center = gray[56:168, 56:168]
    upper_half = gray[:112, :]
    lower_half = gray[112:, :]

    border_std = float(np.std(border) / 255.0)
    center_std = float(np.std(center) / 255.0)
    border_mean = float(np.mean(border) / 255.0)
    white_ratio = float(np.mean(gray > 235))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    upper_edge_density = float(np.mean(cv2.Canny(upper_half, 80, 160) > 0))
    lower_edge_density = float(np.mean(cv2.Canny(lower_half, 80, 160) > 0))

    rgb = resized.astype(np.float32)
    border_rgb = np.concatenate(
        [
            rgb[:24, :, :].reshape(-1, 3),
            rgb[-24:, :, :].reshape(-1, 3),
            rgb[:, :24, :].reshape(-1, 3),
            rgb[:, -24:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    center_rgb = rgb[56:168, 56:168, :].reshape(-1, 3)
    border_rgb_mean = border_rgb.mean(axis=0)
    center_rgb_mean = center_rgb.mean(axis=0)
    color_gap = float(np.linalg.norm(center_rgb_mean - border_rgb_mean) / (255.0 * np.sqrt(3.0)))

    distance_map = np.linalg.norm(rgb - border_rgb_mean.reshape(1, 1, 3), axis=2)
    foreground_mask = distance_map > 45.0
    center_foreground_ratio = float(np.mean(foreground_mask[56:168, 56:168]))
    border_foreground_ratio = float(
        np.mean(
            np.concatenate(
                [
                    foreground_mask[:24, :].ravel(),
                    foreground_mask[-24:, :].ravel(),
                    foreground_mask[:, :24].ravel(),
                    foreground_mask[:, -24:].ravel(),
                ]
            )
        )
    )

    edge_mask = (edges > 0).astype(np.uint8) * 255
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_center_contour_ratio = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = (x + w / 2.0) / 224.0
        cy = (y + h / 2.0) / 224.0
        ratio = area / float(224 * 224)
        if 0.2 <= cx <= 0.8 and 0.2 <= cy <= 0.9:
            largest_center_contour_ratio = max(largest_center_contour_ratio, float(ratio))

    score = 0.0
    if white_ratio > 0.35:
        score += 0.30
    if border_mean > 0.72 and border_std < 0.08:
        score += 0.20
    if center_std > border_std * 1.8:
        score += 0.10
    if edge_density < 0.08:
        score += 0.10
    if color_gap > 0.18:
        score += 0.15
    if center_foreground_ratio > max(0.18, border_foreground_ratio * 1.8):
        score += 0.20
    if lower_edge_density > max(0.12, upper_edge_density * 1.6):
        score += 0.15
    if 0.05 < largest_center_contour_ratio < 0.55:
        score += 0.15

    score = float(np.clip(score, 0.0, 1.0))
    return {
        "score": score,
        "white_ratio": white_ratio,
        "border_mean": border_mean,
        "border_std": border_std,
        "center_std": center_std,
        "edge_density": edge_density,
        "upper_edge_density": upper_edge_density,
        "lower_edge_density": lower_edge_density,
        "color_gap": color_gap,
        "center_foreground_ratio": center_foreground_ratio,
        "border_foreground_ratio": border_foreground_ratio,
        "largest_center_contour_ratio": largest_center_contour_ratio,
        "is_likely_non_satellite": score >= 0.55,
    }


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

    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset de origem não encontrado: {source_dir}")

    source_counts = count_images_by_class(source_dir, classes)
    train_counts = count_images_by_class(train_root, classes)
    val_counts = count_images_by_class(val_root, classes)
    current_split_counts = {
        class_name: train_counts.get(class_name, 0) + val_counts.get(class_name, 0)
        for class_name in classes
    }

    should_rebuild = not (train_root.exists() and val_root.exists())
    if not should_rebuild:
        should_rebuild = any(source_counts[class_name] != current_split_counts[class_name] for class_name in classes)

    if should_rebuild and output_dir.exists():
        shutil.rmtree(output_dir)

    if not should_rebuild:
        return output_dir

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
