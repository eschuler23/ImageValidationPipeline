#!/usr/bin/env python3
"""
Plot per-epoch validation curves for latest 6 SqueezeNet runs + a ResNet50 DTD reference.

Outputs:
- <output-prefix>_f1.png
- <output-prefix>_acc.png

How to run:
  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/squeezenet6_plus_resnet50dtd_curves.py \
    --runs-root Image_Processing/Content_validation/runs \
    --num-squeezenet-runs 6 \
    --comparison-run dtd_resnet50_seed21_ep15_freeze5_val15_test5_20260205 \
    --output-prefix reports/squeezenet6_plus_resnet50dtd_curves \
    --epoch-jitter 0.35
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


@dataclass(frozen=True)
class RunSeries:
    run_name: str
    model: str
    weights: str
    freeze_backbone_epochs: Optional[int]
    epochs: List[int]
    f1: List[float]
    acc: List[float]

    def label(self) -> str:
        freeze_text = "?" if self.freeze_backbone_epochs is None else str(self.freeze_backbone_epochs)
        return f"{self.run_name} | {self.model}/{self.weights} | freeze={freeze_text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latest 6 SqueezeNet runs + ResNet50 DTD comparison."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--num-squeezenet-runs",
        type=int,
        default=6,
        help="How many latest SqueezeNet runs to include.",
    )
    parser.add_argument(
        "--comparison-run",
        type=str,
        default="dtd_resnet50_seed21_ep15_freeze5_val15_test5_20260205",
        help="Run folder name for the ResNet50 DTD comparison series.",
    )
    parser.add_argument(
        "--comparison-run-2",
        type=str,
        default="dtd_resnet50_seed21_ep15_freeze1_20260205",
        help="Second ResNet50 DTD run to include as comparison.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/squeezenet6_plus_resnet50dtd_curves"),
        help="Output prefix; writes '<prefix>_f1.png' and '<prefix>_acc.png'.",
    )
    parser.add_argument(
        "--epoch-jitter",
        type=float,
        default=0.35,
        help="Horizontal epoch jitter for overlapping curves.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_model_dir(run_dir: Path, preferred_model_dir: Optional[str] = None) -> Path:
    if preferred_model_dir:
        candidate = run_dir / preferred_model_dir
        if candidate.exists() and (candidate / "metrics.json").exists():
            return candidate
    model_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and (path / "metrics.json").exists())
    if not model_dirs:
        raise FileNotFoundError(f"No model directory with metrics.json found under {run_dir}")
    if len(model_dirs) > 1:
        names = ", ".join(path.name for path in model_dirs)
        raise ValueError(f"Multiple model dirs under {run_dir}: {names}")
    return model_dirs[0]


def load_run_series(runs_root: Path, run_name: str, preferred_model_dir: Optional[str] = None) -> RunSeries:
    run_dir = runs_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    preprocessing_path = run_dir / "preprocessing.json"
    preprocessing = _load_json(preprocessing_path) if preprocessing_path.exists() else {}
    training = preprocessing.get("training") if isinstance(preprocessing, dict) else {}
    training = training if isinstance(training, dict) else {}

    model_dir = _resolve_model_dir(run_dir, preferred_model_dir=preferred_model_dir)
    metrics = _load_json(model_dir / "metrics.json")
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Empty history for {run_name}")

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
        raise ValueError(f"No usable epoch rows for {run_name}")

    return RunSeries(
        run_name=run_name,
        model=str(metrics.get("model", model_dir.name)),
        weights=str(metrics.get("weights", "unknown")),
        freeze_backbone_epochs=_to_int_or_none(training.get("freeze_backbone_epochs")),
        epochs=epochs,
        f1=f1,
        acc=acc,
    )


def latest_squeezenet_runs(runs_root: Path, limit: int) -> List[str]:
    candidates: List[tuple[float, str]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "squeezenet1_1" / "metrics.json"
        if not metrics_path.exists():
            continue
        candidates.append((run_dir.stat().st_mtime, run_dir.name))
    candidates.sort(reverse=True)
    return [name for _, name in candidates[:limit]]


def _offsets(n: int, jitter: float) -> List[float]:
    if n <= 1:
        return [0.0]
    return [(-jitter + (2.0 * jitter) * idx / (n - 1)) for idx in range(n)]


def plot_metric(
    series: Sequence[RunSeries],
    *,
    metric: str,
    title: str,
    output_path: Path,
    epoch_jitter: float,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    dtd_colors = [
        "#1f77b4",
        "#4a90d9",
        "#0b5fa5",
        "#6ba6de",
    ]
    squeezenet_colors = [
        "#f1c40f",
        "#e0b000",
        "#f5d547",
        "#c79a00",
        "#ffd84d",
        "#d4a300",
        "#e8be2f",
        "#b78c00",
    ]
    fallback_colors = [
        "#9e9e9e",
        "#7f7f7f",
        "#b3b3b3",
        "#666666",
    ]
    offsets = _offsets(len(series), epoch_jitter)
    dtd_idx = 0
    squeezenet_idx = 0
    fallback_idx = 0

    for idx, run in enumerate(series):
        y_vals = run.f1 if metric == "f1" else run.acc
        x_vals = [epoch + offsets[idx] for epoch in run.epochs]
        is_dtd_series = "dtd" in run.run_name.lower() or run.weights.lower() == "dtd"
        if is_dtd_series:
            line_color = dtd_colors[dtd_idx % len(dtd_colors)]
            dtd_idx += 1
        elif run.model.lower() == "squeezenet1_1":
            line_color = squeezenet_colors[squeezenet_idx % len(squeezenet_colors)]
            squeezenet_idx += 1
        else:
            line_color = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
        ax.plot(
            x_vals,
            y_vals,
            color=line_color,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=run.label(),
        )

    max_epoch = max(max(run.epochs) for run in series)
    max_abs_offset = max(abs(offset) for offset in offsets) if offsets else 0.0
    ax.set_xlim(0, max_epoch + max_abs_offset + 0.5)
    ax.set_xticks(list(range(0, max_epoch + 1)))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 (weighted)" if metric == "f1" else "Validation Accuracy")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=8,
        title="Run | Model/Weights | Freeze",
        title_fontsize=9,
        ncol=1,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    squeezenet_runs = latest_squeezenet_runs(args.runs_root, args.num_squeezenet_runs)
    if len(squeezenet_runs) < args.num_squeezenet_runs:
        raise SystemExit(
            f"Requested {args.num_squeezenet_runs} SqueezeNet runs, found only {len(squeezenet_runs)}."
        )

    all_series: List[RunSeries] = []
    for run_name in squeezenet_runs:
        all_series.append(load_run_series(args.runs_root, run_name, preferred_model_dir="squeezenet1_1"))
    comparison_runs: List[str] = [args.comparison_run, args.comparison_run_2]
    unique_comparison_runs: List[str] = []
    for run_name in comparison_runs:
        if run_name and run_name not in unique_comparison_runs:
            unique_comparison_runs.append(run_name)
    for run_name in unique_comparison_runs:
        all_series.append(load_run_series(args.runs_root, run_name, preferred_model_dir="resnet50"))

    f1_path = args.output_prefix.parent / f"{args.output_prefix.name}_f1.png"
    acc_path = args.output_prefix.parent / f"{args.output_prefix.name}_acc.png"

    plot_metric(
        all_series,
        metric="f1",
        title=(
            "Per-epoch Validation F1: 6 latest SqueezeNet runs + "
            "ResNet50 DTD references (freeze5 val15/5 + freeze1 val10/10)"
        ),
        output_path=f1_path,
        epoch_jitter=args.epoch_jitter,
    )
    plot_metric(
        all_series,
        metric="acc",
        title=(
            "Per-epoch Validation Accuracy: 6 latest SqueezeNet runs + "
            "ResNet50 DTD references (freeze5 val15/5 + freeze1 val10/10)"
        ),
        output_path=acc_path,
        epoch_jitter=args.epoch_jitter,
    )

    print("Selected SqueezeNet runs:")
    for run_name in squeezenet_runs:
        print(f"- {run_name}")
    print("Comparison runs:")
    for run_name in unique_comparison_runs:
        print(f"- {run_name}")
    print(f"Wrote: {f1_path}")
    print(f"Wrote: {acc_path}")
    print(f"Series plotted: {len(all_series)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
