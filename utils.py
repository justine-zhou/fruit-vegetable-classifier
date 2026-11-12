from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for utility tests
    np = None


DATA_DIR = Path("frutable")
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
DEFAULT_MODEL_PATH = MODELS_DIR / "best_resnet18.pth"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRUIT_CLASSES = {
    "apple",
    "banana",
    "grapes",
    "kiwi",
    "lemon",
    "mango",
    "orange",
    "pear",
    "pineapple",
    "pomegranate",
    "watermelon",
}

VEGETABLE_CLASSES = {
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "jalepeno",
    "lettuce",
    "onion",
    "paprika",
    "peas",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
}

REQUIRED_SPLITS = ("train", "validation", "test")


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower()


def get_broad_category(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized in FRUIT_CLASSES:
        return "fruit"
    if normalized in VEGETABLE_CLASSES:
        return "vegetable"
    return "unknown"


def ensure_dataset_layout(data_dir: str | Path = DATA_DIR) -> Path:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    missing_splits = [
        split for split in REQUIRED_SPLITS if not (data_path / split).is_dir()
    ]
    if missing_splits:
        joined = ", ".join(missing_splits)
        raise FileNotFoundError(f"Missing dataset split directories: {joined}")

    return data_path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_class_dirs(split_dir: Path) -> Iterable[Path]:
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def summarize_dataset(data_dir: str | Path = DATA_DIR) -> Dict[str, Dict[str, int]]:
    data_path = ensure_dataset_layout(data_dir)
    summary: Dict[str, Dict[str, int]] = {}

    for split in REQUIRED_SPLITS:
        split_dir = data_path / split
        class_dirs = list(iter_class_dirs(split_dir))
        image_count = 0
        for class_dir in class_dirs:
            image_count += sum(1 for path in class_dir.iterdir() if is_image_file(path))
        summary[split] = {"classes": len(class_dirs), "images": image_count}

    return summary


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
