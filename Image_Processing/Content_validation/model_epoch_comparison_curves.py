#!/usr/bin/env python3
"""
Plot per-epoch validation F1 and accuracy curves for selected model runs.

This script is focused on direct run-to-run comparison. It loads each run's:
- model + weights from `*/metrics.json`
- freeze_backbone_epochs from `preprocessing.json`
- per-epoch validation F1/accuracy from `history`

How to run:
  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/model_epoch_comparison_curves.py \
    --runs-root Image_Processing/Content_validation/runs \
    --title-split "val 15% / test 5%" \
    --epoch-jitter 0.18 \
    --output-prefix reports/resnet_model_epoch_comparison_val15_test5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    "dtd_resnet50_seed21_ep15_freeze5_val15_test5_20260205",
    "dtd_resnet50_seed21_ep15_freeze1_20260205",
    "resnet50_imagenet_ep15_freeze5_val15_test5_seed21",
    "resnet50_imagenet_ep20_freeze5_val15_test5_seed21",
    "resnet50_imagenet_ep15_frozen_backbone_val15_test5_seed21",
    "resnet18_imagenet_ep15_freeze5_val15_test5_20260207",
]


@dataclass(frozen=True)
class RunSeries:
    run_name: str
    model: str
    weights: str
    unfreeze_after_epochs: Optional[int]
    epochs: List[int]
    f1: List[float]
    acc: List[float]

    def legend_label(self) -> str:
        freeze_text = "?" if self.unfreeze_after_epochs is None else str(self.unfreeze_after_epochs)
        return f"{self.model} | {self.weights} | unfreeze after {freeze_text} ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected run epoch curves.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root directory that contains run folders.",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run folder names to include in the comparison.",
    )
    parser.add_argument(
        "--title-split",
        type=str,
        default="val 15% / test 5%",
        help="Split text shown in the chart title (not in the legend).",
    )
    parser.add_argument(
        "--comparison-note",
        type=str,
        default="includes DTD freeze=1 comparison (val 10% / test 10%)",
        help="Extra note appended to chart titles.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/resnet_model_epoch_comparison_val15_test5"),
        help="Output prefix. Script writes '<prefix>_f1.png' and '<prefix>_acc.png'.",
    )
    parser.add_argument(
        "--epoch-jitter",
        type=float,
        default=0.18,
        help="Horizontal epoch jitter applied per run to reduce overlap.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_dir(run_dir: Path) -> Path:
    model_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and (path / "metrics.json").exists())
    if not model_dirs:
        raise FileNotFoundError(f"No model directory with metrics.json found under: {run_dir}")
    if len(model_dirs) > 1:
        names = ", ".join(path.name for path in model_dirs)
        raise ValueError(f"Multiple model directories found under {run_dir}: {names}")
    return model_dirs[0]


def _to_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_run_series(runs_root: Path, run_name: str) -> RunSeries:
    run_dir = runs_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    preprocessing_path = run_dir / "preprocessing.json"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"preprocessing.json not found: {preprocessing_path}")
    preprocessing = _load_json(preprocessing_path)

    model_dir = _resolve_model_dir(run_dir)
    metrics_path = model_dir / "metrics.json"
    metrics = _load_json(metrics_path)

    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Empty history in {metrics_path}")

    epochs: List[int] = []
    f1: List[float] = []
    acc: List[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        epoch_raw = row.get("epoch")
        f1_raw = row.get("f1")
        acc_raw = row.get("accuracy")
        if epoch_raw is None or f1_raw is None or acc_raw is None:
            continue
        epochs.append(int(epoch_raw))
        f1.append(float(f1_raw))
        acc.append(float(acc_raw))

    if not epochs:
        raise ValueError(f"No usable epoch rows in {metrics_path}")

    training = preprocessing.get("training") if isinstance(preprocessing, dict) else {}
    training = training if isinstance(training, dict) else {}

    return RunSeries(
        run_name=run_name,
        model=str(metrics.get("model", model_dir.name)),
        weights=str(metrics.get("weights", "unknown")),
        unfreeze_after_epochs=_to_int_or_none(training.get("freeze_backbone_epochs")),
        epochs=epochs,
        f1=f1,
        acc=acc,
    )


def _plot_metric(
    series: Sequence[RunSeries],
    *,
    metric_name: str,
    title_split: str,
    comparison_note: str,
    output_path: Path,
    epoch_jitter: float,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.2))

    dtd_colors = [
        "#1f77b4",
        "#4a90d9",
        "#0b5fa5",
        "#6ba6de",
    ]
    resnet18_colors = [
        "#2ca02c",
        "#6cbf6c",
        "#1e7f1e",
        "#45a145",
    ]
    resnet50_imagenet_colors = [
        "#d62728",
        "#e05a5b",
        "#a81d1e",
        "#ef7a7b",
    ]
    squeezenet_colors = [
        "#f1c40f",
        "#e0b000",
        "#f5d547",
        "#c79a00",
    ]
    fallback_colors = [
        "#9e9e9e",
        "#7f7f7f",
        "#b3b3b3",
        "#666666",
        "#8c8c8c",
        "#5f5f5f",
        "#a8a8a8",
        "#747474",
    ]

    if len(series) == 1:
        offsets = [0.0]
    else:
        offsets = [
            -epoch_jitter + (2.0 * epoch_jitter) * idx / (len(series) - 1)
            for idx in range(len(series))
        ]

    dtd_idx = 0
    resnet18_idx = 0
    resnet50_imagenet_idx = 0
    squeezenet_idx = 0
    fallback_idx = 0
    for idx, run in enumerate(series):
        y_vals = run.f1 if metric_name == "f1" else run.acc
        x_vals = [epoch + offsets[idx] for epoch in run.epochs]
        model = run.model.lower()
        weights = run.weights.lower()
        is_dtd = weights == "dtd" or "dtd" in run.run_name.lower()
        if is_dtd:
            line_color = dtd_colors[dtd_idx % len(dtd_colors)]
            dtd_idx += 1
        elif model == "resnet18":
            line_color = resnet18_colors[resnet18_idx % len(resnet18_colors)]
            resnet18_idx += 1
        elif model == "resnet50" and weights == "imagenet":
            line_color = resnet50_imagenet_colors[
                resnet50_imagenet_idx % len(resnet50_imagenet_colors)
            ]
            resnet50_imagenet_idx += 1
        elif model == "squeezenet1_1":
            line_color = squeezenet_colors[squeezenet_idx % len(squeezenet_colors)]
            squeezenet_idx += 1
        else:
            line_color = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
        ax.plot(
            x_vals,
            y_vals,
            label=run.legend_label(),
            color=line_color,
            marker="o",
            linewidth=1.8,
            markersize=4,
        )

    max_epoch = max(max(run.epochs) for run in series)
    ax.set_xticks(list(range(0, max_epoch + 1)))
    max_abs_offset = max(abs(offset) for offset in offsets) if offsets else 0.0
    ax.set_xlim(0, max_epoch + max_abs_offset + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Epoch")
    title_suffix = f" | {comparison_note}" if comparison_note.strip() else ""
    if metric_name == "f1":
        ax.set_ylabel("Validation F1 (weighted)")
        ax.set_title(f"Validation F1 per Epoch | split: {title_split}{title_suffix}")
    else:
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"Validation Accuracy per Epoch | split: {title_split}{title_suffix}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9,
        title="Model | Weights | Backbone Unfreeze",
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    loaded_series: List[RunSeries] = []
    for run_name in args.run_names:
        loaded_series.append(load_run_series(args.runs_root, run_name))

    output_prefix = args.output_prefix
    f1_path = output_prefix.parent / f"{output_prefix.name}_f1.png"
    acc_path = output_prefix.parent / f"{output_prefix.name}_acc.png"

    _plot_metric(
        loaded_series,
        metric_name="f1",
        title_split=args.title_split,
        comparison_note=args.comparison_note,
        output_path=f1_path,
        epoch_jitter=args.epoch_jitter,
    )
    _plot_metric(
        loaded_series,
        metric_name="acc",
        title_split=args.title_split,
        comparison_note=args.comparison_note,
        output_path=acc_path,
        epoch_jitter=args.epoch_jitter,
    )

    print(f"Wrote: {f1_path}")
    print(f"Wrote: {acc_path}")
    print(f"Runs plotted: {len(loaded_series)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
