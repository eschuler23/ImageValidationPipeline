#!/usr/bin/env python3
"""
Run inference on a folder of images using a trained content-validation model.

How to run:
  uv run python Image_Processing/Content_validation/predict_folder.py \
    --run-dir Image_Processing/Content_validation/runs/run_20260127_005831 \
    --image-dir /path/to/images \
    --output-csv reports/folder_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import train_binary_classifier as tbc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PredictionRow:
    image_path: str
    filename: str
    probability: float
    predicted_label: str
    predicted_binary: int


class FolderImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Path]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, path


def list_images(image_dir: Path, recursive: bool) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image dir is not a directory: {image_dir}")

    iterator: Iterable[Path] = image_dir.rglob("*") if recursive else image_dir.iterdir()
    images = [
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No images found in {image_dir}.")
    return sorted(images)


def find_model_dir(run_dir: Path, model_name: Optional[str]) -> Path:
    if model_name:
        candidate = run_dir / model_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model directory not found: {candidate}")

    model_dirs = [path for path in run_dir.iterdir() if path.is_dir() and (path / "best_model.pt").exists()]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories with best_model.pt under {run_dir}")
    if len(model_dirs) > 1:
        names = ", ".join(sorted(path.name for path in model_dirs))
        raise ValueError(f"Multiple model directories found ({names}); pass --model-name.")
    return model_dirs[0]


def load_preprocessing(run_dir: Path) -> Tuple[int, Sequence[str], Sequence[str]]:
    preprocessing_path = run_dir / "preprocessing.json"
    if not preprocessing_path.exists():
        return 224, ["usable"], ["not usable"]
    payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    image_size = int(payload.get("image_size", 224))
    positive_labels = payload.get("positive_labels") or ["usable"]
    negative_labels = payload.get("negative_labels") or ["not usable"]
    return image_size, positive_labels, negative_labels


def load_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, dict[int, str]]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {model_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = metrics.get("model", model_dir.name)
    weights = metrics.get("weights", "none")

    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {model_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)

    model = tbc.build_model(model_name, weights)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    run_dir = model_dir.parent
    _, positive_labels, negative_labels = load_preprocessing(run_dir)
    label_texts = tbc.build_label_texts(positive_labels, negative_labels)
    return model, label_texts


def predict_folder(
    *,
    run_dir: Path,
    model_name: Optional[str],
    image_dir: Path,
    output_csv: Path,
    device: torch.device,
    batch_size: int,
    threshold: float,
    recursive: bool,
) -> None:
    model_dir = find_model_dir(run_dir, model_name)
    image_size, _, _ = load_preprocessing(run_dir)
    transform = tbc.build_transforms(
        image_size=image_size,
        random_augment=False,
        crop_jitter=False,
        jpeg_quality=None,
        jpeg_prob=0.0,
        noise_std=None,
        noise_prob=0.0,
        color_jitter=False,
        color_jitter_hue=0.0,
        color_jitter_saturation=0.0,
        color_jitter_contrast=0.0,
        brightness_jitter=False,
        brightness_jitter_delta=0.0,
        random_hflip=False,
        random_vflip=False,
        random_rotations=(),
        random_rotation_prob=0.0,
    )

    images = list_images(image_dir, recursive=recursive)
    dataset = FolderImageDataset(images, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, label_texts = load_model(model_dir, device)

    rows: List[PredictionRow] = []
    with torch.no_grad():
        for batch_images, batch_paths in loader:
            batch_images = batch_images.to(device)
            logits = model(batch_images).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            for path, prob in zip(batch_paths, probs):
                filename = path.name
                pred_binary = 1 if prob >= threshold else 0
                label = label_texts.get(pred_binary, str(pred_binary))
                rows.append(
                    PredictionRow(
                        image_path=str(path),
                        filename=filename,
                        probability=float(prob),
                        predicted_label=label,
                        predicted_binary=pred_binary,
                    )
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "filename",
                "probability",
                "predicted_label",
                "predicted_binary",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    positives = sum(row.predicted_binary for row in rows)
    negatives = len(rows) - positives
    print(f"Saved predictions to {output_csv}")
    print(f"Total images: {len(rows)} | predicted usable: {positives} | predicted not usable: {negatives}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on a folder of images.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing the model.")
    parser.add_argument("--model-name", type=str, help="Model subdirectory name (if run has multiple models).")
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder of images to score.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path for predictions.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive label.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = tbc.resolve_device(args.device)
    predict_folder(
        run_dir=args.run_dir,
        model_name=args.model_name,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
