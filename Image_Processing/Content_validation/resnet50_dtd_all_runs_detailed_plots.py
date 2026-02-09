#!/usr/bin/env python3
"""
Plot all ResNet50 DTD runs with run-specific configuration labels.

This script scans `runs/` for runs where:
- model == resnet50
- weights == dtd

For each run, it plots per-epoch validation metrics and includes run-specific
details directly in the legend label:
- planned epochs
- seed
- split (val/test)
- freeze_backbone_epochs
- random augmentation flag

Two output PNG files are written (one metric per file):
- <output-prefix>_f1.png
- <output-prefix>_acc.png

How to run:
  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/resnet50_dtd_all_runs_detailed_plots.py \
    --runs-root Image_Processing/Content_validation/runs \
    --output-prefix reports/resnet50_dtd_all_runs_detailed \
    --epoch-jitter 0.18
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED_RE = re.compile(r"(?:^|[_-])seed[-_]?([0-9]+)", re.IGNORECASE)
FREEZE_RE = re.compile(r"(?:^|[_-])freeze[-_]?([0-9]+)", re.IGNORECASE)
EPOCH_RE = re.compile(r"(?:^|[_-])ep(?:ochs?)?[-_]?([0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class RunSeries:
    run_name: str
    epochs: List[int]
    f1: List[float]
    acc: List[float]
    epochs_planned: Optional[int]
    seed: Optional[int]
    freeze_backbone_epochs: Optional[int]
    val_pct: Optional[float]
    test_pct: Optional[float]
    val_count: Optional[int]
    test_count: Optional[int]
    random_augmentation: Optional[bool]

    def label(self) -> str:
        ep_text = str(self.epochs_planned) if self.epochs_planned is not None else "?"
        seed_text = str(self.seed) if self.seed is not None else "?"
        freeze_text = str(self.freeze_backbone_epochs) if self.freeze_backbone_epochs is not None else "?"
        aug_text = (
            "on" if self.random_augmentation is True else "off" if self.random_augmentation is False else "?"
        )
        if self.val_pct is None or self.test_pct is None:
            split_text = "?/?"
        else:
            split_text = f"{self.val_pct:.2f}/{self.test_pct:.2f}"
        if self.val_count is None or self.test_count is None:
            split_counts_text = "?/?"
        else:
            split_counts_text = f"{self.val_count}/{self.test_count}"
        return (
            f"{self.run_name} | ep={ep_text} seed={seed_text} "
            f"split={split_text} (v/t={split_counts_text}) "
            f"freeze={freeze_text} aug={aug_text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all ResNet50 DTD runs with detailed labels.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/resnet50_dtd_all_runs_detailed"),
        help="Output prefix; writes '<prefix>_f1.png' and '<prefix>_acc.png'.",
    )
    parser.add_argument(
        "--epoch-jitter",
        type=float,
        default=0.18,
        help="Horizontal jitter to separate overlapping lines.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_seed(run_name: str) -> Optional[int]:
    match = SEED_RE.search(run_name)
    if not match:
        return None
    return _parse_int(match.group(1))


def _infer_freeze(run_name: str) -> Optional[int]:
    match = FREEZE_RE.search(run_name)
    if not match:
        return None
    return _parse_int(match.group(1))


def _infer_epochs(run_name: str) -> Optional[int]:
    match = EPOCH_RE.search(run_name)
    if not match:
        return None
    return _parse_int(match.group(1))


def _extract_split(preprocessing: dict) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    splits = preprocessing.get("splits")
    if not isinstance(splits, dict):
        return None, None, None, None
    train = _parse_int(splits.get("train_base"))
    val = _parse_int(splits.get("val_base"))
    test = _parse_int(splits.get("test_base"))
    if train is None or val is None or test is None:
        return None, None, val, test
    total = train + val + test
    if total <= 0:
        return None, None, val, test
    return val / total, test / total, val, test


def _extract_aug(preprocessing: dict) -> Optional[bool]:
    if not isinstance(preprocessing, dict):
        return None
    if "random_augmentation" not in preprocessing:
        return None
    value = preprocessing.get("random_augmentation")
    if isinstance(value, bool):
        return value
    return None


def _resolve_metadata(run_name: str, preprocessing: dict, history_len: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    training = preprocessing.get("training")
    training = training if isinstance(training, dict) else {}

    seed = _parse_int(training.get("seed"))
    if seed is None:
        seed = _infer_seed(run_name)

    epochs_planned = _parse_int(training.get("epochs"))
    if epochs_planned is None:
        epochs_planned = _infer_epochs(run_name)
    if epochs_planned is None:
        epochs_planned = history_len

    freeze = _parse_int(training.get("freeze_backbone_epochs"))
    if freeze is None:
        freeze = _infer_freeze(run_name)
    if freeze is None and "frozen_backbone" in run_name.lower():
        freeze = epochs_planned

    return epochs_planned, seed, freeze


def load_resnet50_dtd_runs(runs_root: Path) -> List[RunSeries]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    series: List[RunSeries] = []
    for metrics_path in sorted(runs_root.rglob("resnet50/metrics.json")):
        metrics = _load_json(metrics_path)
        if metrics.get("model") != "resnet50":
            continue
        if metrics.get("weights") != "dtd":
            continue

        history = metrics.get("history")
        if not isinstance(history, list) or not history:
            continue

        run_dir = metrics_path.parent.parent
        run_name = run_dir.name

        preprocessing_path = run_dir / "preprocessing.json"
        preprocessing = _load_json(preprocessing_path) if preprocessing_path.exists() else {}

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
            continue

        val_pct, test_pct, val_count, test_count = _extract_split(preprocessing)
        epochs_planned, seed, freeze = _resolve_metadata(run_name, preprocessing, len(epochs))
        random_aug = _extract_aug(preprocessing)

        series.append(
            RunSeries(
                run_name=run_name,
                epochs=epochs,
                f1=f1,
                acc=acc,
                epochs_planned=epochs_planned,
                seed=seed,
                freeze_backbone_epochs=freeze,
                val_pct=val_pct,
                test_pct=test_pct,
                val_count=val_count,
                test_count=test_count,
                random_augmentation=random_aug,
            )
        )

    series.sort(key=lambda item: item.run_name)
    return series


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
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    colors = [
        "#1f77b4",
        "#4a90d9",
        "#0b5fa5",
        "#6ba6de",
        "#2f7fbe",
        "#8bbce5",
        "#165a8f",
        "#3a86c8",
        "#5a9ed8",
        "#0a4f86",
    ]

    offsets = _offsets(len(series), epoch_jitter)
    for idx, run in enumerate(series):
        y_vals = run.f1 if metric == "f1" else run.acc
        x_vals = [epoch + offsets[idx] for epoch in run.epochs]
        ax.plot(
            x_vals,
            y_vals,
            color=colors[idx % len(colors)],
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            label=run.label(),
        )

    max_epoch = max(max(run.epochs) for run in series)
    max_abs_offset = max(abs(offset) for offset in offsets) if offsets else 0.0
    ax.set_xlim(0, max_epoch + max_abs_offset + 0.5)
    ax.set_xticks(list(range(0, max_epoch + 1)))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 (weighted)" if metric == "f1" else "Validation Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="run | ep seed split freeze aug",
        title_fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.73, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    series = load_resnet50_dtd_runs(args.runs_root)
    if not series:
        raise SystemExit("No ResNet50 DTD runs found.")

    f1_path = args.output_prefix.parent / f"{args.output_prefix.name}_f1.png"
    acc_path = args.output_prefix.parent / f"{args.output_prefix.name}_acc.png"

    plot_metric(
        series,
        metric="f1",
        title="All ResNet50 DTD Runs: Validation F1 per Epoch",
        output_path=f1_path,
        epoch_jitter=args.epoch_jitter,
    )
    plot_metric(
        series,
        metric="acc",
        title="All ResNet50 DTD Runs: Validation Accuracy per Epoch",
        output_path=acc_path,
        epoch_jitter=args.epoch_jitter,
    )

    print(f"Wrote: {f1_path}")
    print(f"Wrote: {acc_path}")
    print(f"Runs plotted: {len(series)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
