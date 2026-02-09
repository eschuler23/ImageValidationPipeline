#!/usr/bin/env python3
"""
Render a grid of misclassified images (FP/FN) with labels.

How to run:
  uv run python Image_Processing/Content_validation/misclassification_grid.py \
    --run-dir Image_Processing/Content_validation/runs/dtd_resnet50_detblur_ep15 \
    --image-dir Imagedump271 \
    --labels-csv AURA271labels.csv \
    --output-path ~/Downloads/aura271_misclassifications.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import train_binary_classifier as tbc


@dataclass(frozen=True)
class Misclassification:
    image_path: Path
    true_label: int
    pred_label: int
    probability: float
    label_text: str
    pred_text: str
    extra_labels: List[Tuple[str, str]]


class FolderDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, path.name


def load_preprocessing(run_dir: Path) -> Tuple[int, Sequence[str], Sequence[str]]:
    preprocessing_path = run_dir / "preprocessing.json"
    if preprocessing_path.exists():
        payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        image_size = int(payload.get("image_size", 224))
        positive_labels = payload.get("positive_labels") or ["usable"]
        negative_labels = payload.get("negative_labels") or ["not usable"]
        return image_size, positive_labels, negative_labels
    return 224, ["usable"], ["not usable"]


def resolve_model_dir(run_dir: Path, model_name: Optional[str]) -> Path:
    if model_name:
        model_dir = run_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir
    candidates = [path for path in run_dir.iterdir() if path.is_dir() and (path / "best_model.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No model directory with best_model.pt found under {run_dir}")
    if len(candidates) > 1:
        names = ", ".join(sorted(path.name for path in candidates))
        raise ValueError(f"Multiple model dirs found ({names}); pass --model-name.")
    return candidates[0]


def load_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, dict[int, str]]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {model_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = metrics.get("model", model_dir.name)
    weights = metrics.get("weights", "none")

    checkpoint_path = model_dir / "best_model.pt"
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


def build_eval_transform(image_size: int):
    return tbc.build_transforms(
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


def render_grid(
    misclassified: List[Misclassification],
    output_path: Path,
    *,
    columns: int,
    image_size: int,
) -> None:
    if not misclassified:
        raise ValueError("No misclassified images found.")

    total = len(misclassified)
    cols = max(1, columns)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.8))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.ravel().tolist()

    for idx, item in enumerate(misclassified):
        ax = axes_list[idx]
        with Image.open(item.image_path) as img:
            img = img.convert("RGB")
            img = img.resize((image_size, image_size))
        ax.imshow(img)
        ax.axis("off")

        title_lines = [
            f"{item.label_text} -> {item.pred_text}",
            f"p={item.probability:.3f}",
        ]
        for key, value in item.extra_labels:
            title_lines.append(f"{key}: {value}")
        ax.set_title("\n".join(title_lines), fontsize=8)

    for ax in axes_list[total:]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a grid of misclassified images.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, help="Model subdir (if run has multiple models).")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument(
        "--label-column",
        type=str,
        default="usability considering nfp",
        help="Primary label column to evaluate.",
    )
    parser.add_argument("--positive-label", type=str, default="usable")
    parser.add_argument("--negative-label", type=str, default="not usable")
    parser.add_argument(
        "--extra-label-columns",
        nargs="*",
        default=["Blurr"],
        help="Extra label columns to display per image.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.home() / "Downloads" / "aura271_misclassifications.png",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = args.run_dir
    model_dir = resolve_model_dir(run_dir, args.model_name)
    image_size, _, _ = load_preprocessing(run_dir)
    transform = build_eval_transform(image_size)

    device = tbc.resolve_device(args.device)
    model, label_texts = load_model(model_dir, device)

    labels_df = pd.read_csv(args.labels_csv)
    labels_df["filename"] = labels_df["filename"].astype(str).str.strip()
    labels_df["label_raw"] = labels_df[args.label_column].astype(str).str.strip().str.lower()

    label_map = {}
    for _, row in labels_df.iterrows():
        filename = row["filename"]
        raw_label = row["label_raw"]
        if raw_label == args.positive_label:
            label_map[filename] = 1
        elif raw_label == args.negative_label:
            label_map[filename] = 0

    image_paths = [p for p in args.image_dir.iterdir() if p.is_file() and p.name in label_map]
    if not image_paths:
        raise SystemExit("No images matched the provided labels.")

    loader = DataLoader(
        FolderDataset(sorted(image_paths), transform),
        batch_size=args.batch_size,
        shuffle=False,
    )

    preds = {}
    probs_map = {}
    with torch.no_grad():
        for batch_images, batch_names in loader:
            batch_images = batch_images.to(device)
            logits = model(batch_images).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            for name, prob in zip(batch_names, probs):
                pred = 1 if prob >= args.threshold else 0
                preds[name] = pred
                probs_map[name] = float(prob)

    misclassified: List[Misclassification] = []
    for _, row in labels_df.iterrows():
        filename = row["filename"]
        if filename not in label_map or filename not in preds:
            continue
        true_label = label_map[filename]
        pred_label = preds[filename]
        if pred_label == true_label:
            continue
        extra_labels = []
        for col in args.extra_label_columns:
            if col in row:
                value = row.get(col, "")
                value = "missing" if value is None or str(value).strip() == "" else str(value)
                extra_labels.append((col, value))
        label_text = label_texts.get(true_label, str(true_label))
        pred_text = label_texts.get(pred_label, str(pred_label))
        misclassified.append(
            Misclassification(
                image_path=args.image_dir / filename,
                true_label=true_label,
                pred_label=pred_label,
                probability=probs_map.get(filename, 0.0),
                label_text=label_text,
                pred_text=pred_text,
                extra_labels=extra_labels,
            )
        )

    render_grid(misclassified, args.output_path, columns=args.columns, image_size=image_size)
    print(f"Saved misclassification grid to {args.output_path}")
    print(f"Total misclassifications: {len(misclassified)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
