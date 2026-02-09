#!/usr/bin/env python3
"""
Generate AURA271 model comparison charts (misclassified counts + confusion bars).

How to run:
  uv run python Image_Processing/Content_validation/aura271_model_comparison_chart.py \
    --runs Image_Processing/Content_validation/runs/dtd_resnet50_detblur_ep15 \
           Image_Processing/Content_validation/runs/dtd_resnet50_seed21_ep11 \
           Image_Processing/Content_validation/runs/run_20260126_164452 \
           Image_Processing/Content_validation/runs/run_20260126_181456 \
    --image-dir Imagedump271 \
    --labels-csv AURA271labels.csv \
    --output-path ~/Downloads/aura271_model_comparison.png
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path("Image_Processing/Content_validation").resolve()))
import train_binary_classifier as tbc


@dataclass(frozen=True)
class ConfusionStats:
    label: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def misclassified(self) -> int:
        return self.fp + self.fn


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


def load_preprocessing(run_dir: Path) -> int:
    preprocessing_path = run_dir / "preprocessing.json"
    if preprocessing_path.exists():
        payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        return int(payload.get("image_size", 224))
    return 224


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


def load_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, str, str]:
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
    return model, model_name, weights


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


def build_label(run_dir: Path, model_name: str, weights: str) -> str:
    return f"{model_name} ({weights})\\n{run_dir.name}"


def evaluate_run(
    *,
    run_dir: Path,
    model_name_override: Optional[str],
    image_dir: Path,
    labels_df: pd.DataFrame,
    label_column: str,
    positive_label: str,
    negative_label: str,
    threshold: float,
    device: torch.device,
) -> ConfusionStats:
    model_dir = resolve_model_dir(run_dir, model_name_override)
    image_size = load_preprocessing(run_dir)
    transform = build_eval_transform(image_size)

    model, model_name, weights = load_model(model_dir, device)

    label_map = {}
    for _, row in labels_df.iterrows():
        filename = row["filename"]
        raw_label = row["label_raw"]
        if raw_label == positive_label:
            label_map[filename] = 1
        elif raw_label == negative_label:
            label_map[filename] = 0

    image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.name in label_map]
    if not image_paths:
        raise ValueError(f"No images matched labels for run {run_dir}")

    loader = DataLoader(FolderDataset(sorted(image_paths), transform), batch_size=16, shuffle=False)

    preds = {}
    with torch.no_grad():
        for batch_images, batch_names in loader:
            batch_images = batch_images.to(device)
            logits = model(batch_images).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            for name, prob in zip(batch_names, probs):
                preds[name] = 1 if prob >= threshold else 0

    tp = fp = fn = tn = 0
    for name, label in label_map.items():
        pred = preds.get(name)
        if pred is None:
            continue
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1

    label = build_label(run_dir, model_name, weights)
    return ConfusionStats(label=label, tp=tp, fp=fp, fn=fn, tn=tn)


def plot_confusion_bars(stats: List[ConfusionStats], output_path: Path) -> None:
    labels = [entry.label for entry in stats]
    x = list(range(len(labels)))

    tp = [entry.tp for entry in stats]
    fp = [entry.fp for entry in stats]
    fn = [entry.fn for entry in stats]
    tn = [entry.tn for entry in stats]
    misclassified = [entry.misclassified for entry in stats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x, misclassified, color="#d95f02")
    axes[0].set_title("Misclassified count (FP + FN)")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    width = 0.2
    axes[1].bar([i - 1.5 * width for i in x], tp, width=width, label="TP")
    axes[1].bar([i - 0.5 * width for i in x], fp, width=width, label="FP")
    axes[1].bar([i + 0.5 * width for i in x], fn, width=width, label="FN")
    axes[1].bar([i + 1.5 * width for i in x], tn, width=width, label="TN")
    axes[1].set_title("Confusion matrix counts")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AURA271 model comparison chart.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="usability considering nfp")
    parser.add_argument("--positive-label", type=str, default="usable")
    parser.add_argument("--negative-label", type=str, default="not usable")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.home() / "Downloads" / "aura271_model_comparison.png",
    )
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)
    labels_df["filename"] = labels_df["filename"].astype(str).str.strip()
    labels_df["label_raw"] = labels_df[args.label_column].astype(str).str.strip().str.lower()

    device = tbc.resolve_device(args.device)

    stats: List[ConfusionStats] = []
    for run_dir in args.runs:
        stats.append(
            evaluate_run(
                run_dir=run_dir,
                model_name_override=None,
                image_dir=args.image_dir,
                labels_df=labels_df,
                label_column=args.label_column,
                positive_label=args.positive_label,
                negative_label=args.negative_label,
                threshold=args.threshold,
                device=device,
            )
        )

    plot_confusion_bars(stats, args.output_path)
    print(f"Saved chart to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
