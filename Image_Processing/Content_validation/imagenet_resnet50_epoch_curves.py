#!/usr/bin/env python3
"""
Plot per-epoch validation F1 and accuracy for ResNet50 ImageNet runs.

This scans the runs folder for metrics.json files, filters to ResNet50 runs
initialized with ImageNet weights, and plots validation F1/accuracy vs epoch.

How to run:
  uv run python Image_Processing/Content_validation/imagenet_resnet50_epoch_curves.py \
    --runs-root Image_Processing/Content_validation/runs \
    --output-path reports/resnet50_imagenet_epoch_curves.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSeries:
    name: str
    model: str
    weights: str
    epochs: List[int]
    f1: List[float]
    acc: List[float]
    freeze_epochs: Optional[int]
    val_pct: Optional[float]
    test_pct: Optional[float]

    def label(self) -> str:
        freeze = "?" if self.freeze_epochs is None else str(self.freeze_epochs)
        if self.val_pct is None or self.test_pct is None:
            split = "val/test=?/?"
        else:
            split = f"val/test={self.val_pct:.2f}/{self.test_pct:.2f}"
        return f"{self.name} | {self.model}/{self.weights} | freeze={freeze} | {split}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-epoch validation curves for ResNet50 ImageNet runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root folder containing training runs.",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="resnet",
        help="Only include models whose name starts with this prefix (default: resnet).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Optional weights filter (e.g., imagenet, dtd).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/resnet50_imagenet_epoch_curves.png"),
        help="Path to write the output PNG.",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Optional YYYY-MM-DD to include only runs from that date (by metrics.json mtime).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_split(preprocessing: dict) -> Tuple[Optional[float], Optional[float]]:
    splits = preprocessing.get("splits") or {}
    train = splits.get("train_base")
    val = splits.get("val_base")
    test = splits.get("test_base")
    if train is None or val is None or test is None:
        return None, None
    total = train + val + test
    if not total:
        return None, None
    return val / total, test / total


def _extract_freeze(preprocessing: dict) -> Optional[int]:
    training = preprocessing.get("training")
    if not isinstance(training, dict):
        return None
    value = training.get("freeze_backbone_epochs")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_resnet_runs(
    runs_root: Path,
    *,
    target_date: Optional[date],
    model_prefix: str,
    weights_filter: Optional[str],
) -> List[RunSeries]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    series: List[RunSeries] = []
    for metrics_path in runs_root.rglob("metrics.json"):
        metrics = _load_json(metrics_path)
        model_name = metrics.get("model", "")
        weights = metrics.get("weights", "")
        if not model_name.startswith(model_prefix):
            continue
        if weights_filter and weights != weights_filter:
            continue
        if target_date:
            run_date = datetime.fromtimestamp(metrics_path.stat().st_mtime).date()
            if run_date != target_date:
                continue

        history = metrics.get("history") or []
        if not history:
            continue

        epochs = [int(row.get("epoch")) for row in history]
        f1 = [float(row.get("f1", 0.0)) for row in history]
        acc = [float(row.get("accuracy", 0.0)) for row in history]

        run_dir = metrics_path.parent.parent
        preprocessing_path = run_dir / "preprocessing.json"
        preprocessing = _load_json(preprocessing_path) if preprocessing_path.exists() else {}
        val_pct, test_pct = _extract_split(preprocessing)
        freeze_epochs = _extract_freeze(preprocessing)

        series.append(
            RunSeries(
                name=run_dir.name,
                model=model_name,
                weights=weights,
                epochs=epochs,
                f1=f1,
                acc=acc,
                freeze_epochs=freeze_epochs,
                val_pct=val_pct,
                test_pct=test_pct,
            )
        )

    series.sort(key=lambda item: item.name)
    return series


def plot_series(series: List[RunSeries], output_path: Path) -> None:
    if not series:
        raise ValueError("No ResNet50 ImageNet runs found.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    ax_f1, ax_acc = axes

    # Explicit palette with a visible brown included.
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#17becf",  # teal
        "#e377c2",  # pink
        "#bcbd22",  # olive
        "#7f7f7f",  # gray
    ]
    for idx, run in enumerate(series):
        color = colors[idx % len(colors)]
        label = run.label()
        ax_f1.plot(run.epochs, run.f1, marker="o", linewidth=1.8, color=color, label=label)
        ax_acc.plot(run.epochs, run.acc, marker="o", linewidth=1.8, color=color)

    ax_f1.set_title("ResNet Validation F1 per Epoch (weighted)")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Val F1")
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.grid(True, alpha=0.3)

    ax_acc.set_title("ResNet Validation Accuracy per Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)

    handles, labels = ax_f1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    target_date = _parse_date(args.date)
    series = load_resnet_runs(
        args.runs_root,
        target_date=target_date,
        model_prefix=args.model_prefix,
        weights_filter=args.weights,
    )
    if not series:
        date_note = args.date or "any date"
        raise SystemExit(f"No matching ResNet50 ImageNet runs found for {date_note}.")
    plot_series(series, args.output_path)
    print(f"Wrote: {args.output_path}")
    print(f"Runs plotted: {len(series)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
