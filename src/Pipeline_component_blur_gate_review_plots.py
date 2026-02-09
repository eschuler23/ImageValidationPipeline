#!/usr/bin/env python3
"""
Generate review plots for the blur gate baseline.

How to run (from repo root):
  uv run python src/Pipeline_component_blur_gate_review_plots.py \
    --predictions-csv reports/baseline_fullimage_predictions.csv \
    --output-dir "specs/Blurr_detection - SAM foreground/artifacts"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve


LABEL_POSITIVE = "too_blurry"
LABEL_NEGATIVE = "focused_enough"


def label_to_binary(label: str) -> int | None:
    label = (label or "").strip().lower()
    if label == LABEL_POSITIVE:
        return 1
    if label == LABEL_NEGATIVE:
        return 0
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_title_footer(ax: plt.Axes, title: str, footer: str | None = None) -> None:
    ax.set_title(title)
    if footer:
        ax.text(
            0.99,
            0.01,
            footer,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )


def plot_histogram_by_label(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for label, color in [(LABEL_POSITIVE, "#e45756"), (LABEL_NEGATIVE, "#54a24b")]:
        subset = df[df["ground_truth_label"] == label]["laplacian_variance"]
        values = pd.to_numeric(subset, errors="coerce").dropna().to_numpy(dtype=float)
        values = values[values > 0]
        if values.size == 0:
            continue
        logv = np.log10(values)
        ax.hist(
            logv,
            bins=40,
            alpha=0.6,
            color=color,
            label=f"{label} (n={values.size})",
        )

    ax.set_xlabel("log10(Laplacian variance)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)
    add_title_footer(ax, "Baseline: Laplacian variance distribution by label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_boxplot_by_label(
    df: pd.DataFrame,
    out_path: Path,
    *,
    log_scale: bool = True,
    show_fliers: bool = False,
    y_max: float | None = None,
    y_cap: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [LABEL_POSITIVE, LABEL_NEGATIVE]
    data = []
    for label in labels:
        values = pd.to_numeric(
            df.loc[df["ground_truth_label"] == label, "laplacian_variance"],
            errors="coerce",
        ).dropna()
        if log_scale:
            values = values[values > 0]
        else:
            values = values[values >= 0]
        data.append(values)

    all_values = pd.concat(data, ignore_index=True) if data else pd.Series([], dtype=float)

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=show_fliers,
    )
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Laplacian variance (log scale)")
        title = "Baseline: Laplacian variance by label"
    else:
        ax.set_ylabel("Laplacian variance")
        title = "Baseline: Laplacian variance by label (linear scale)"
    if show_fliers:
        title = f"{title} (with outliers)"

    axis_max = None
    if y_max is not None:
        axis_max = float(y_max)
    elif y_cap is not None and not all_values.empty:
        capped = all_values[all_values <= y_cap]
        if not capped.empty:
            axis_max = float(capped.max()) * 1.02

    if axis_max is not None:
        ax.set_ylim(top=axis_max)
    if not log_scale:
        ax.set_ylim(bottom=0.0)
    add_title_footer(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr_curve(df: pd.DataFrame, out_path: Path) -> None:
    labeled = df[df["ground_truth_label"].isin([LABEL_POSITIVE, LABEL_NEGATIVE])].copy()
    y_true = labeled["ground_truth_label"].map(label_to_binary)
    y_score = pd.to_numeric(labeled["blur_score"], errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if y_true.nunique() < 2:
        ax.text(
            0.5,
            0.5,
            "PR curve unavailable (single class)",
            ha="center",
            va="center",
        )
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, color="#4c78a8", linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        add_title_footer(ax, "Baseline: Precision-Recall curve")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(df: pd.DataFrame, out_path: Path) -> None:
    labeled = df[df["ground_truth_label"].isin([LABEL_POSITIVE, LABEL_NEGATIVE])].copy()
    y_true = labeled["ground_truth_label"].map(
        {LABEL_POSITIVE: "blurry", LABEL_NEGATIVE: "not_blurry"}
    ).tolist()
    y_pred = labeled["prediction_label"].tolist()

    labels = ["blurry", "not_blurry"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    add_title_footer(ax, "Baseline: Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#1f1f1f")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ground_truth_label" not in df.columns:
        raise ValueError("Missing ground_truth_label in predictions CSV.")
    if "laplacian_variance" not in df.columns:
        raise ValueError("Missing laplacian_variance in predictions CSV.")
    if "blur_score" not in df.columns:
        raise ValueError("Missing blur_score in predictions CSV.")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate baseline review plots.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Path to baseline predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save plot images.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    df = load_predictions(args.predictions_csv)
    ensure_dir(args.output_dir)

    plots = {
        "baseline_variance_hist.png": plot_histogram_by_label,
        "baseline_variance_boxplot.png": plot_boxplot_by_label,
        "baseline_variance_boxplot_linear.png": lambda df, out_path: plot_boxplot_by_label(
            df, out_path, log_scale=False
        ),
        "baseline_variance_boxplot_linear_with_outliers.png": lambda df, out_path: plot_boxplot_by_label(
            df, out_path, log_scale=False, show_fliers=True, y_cap=140000
        ),
        "baseline_pr_curve.png": plot_pr_curve,
        "baseline_confusion_matrix.png": plot_confusion_matrix,
    }

    for filename, fn in plots.items():
        out_path = args.output_dir / filename
        fn(df, out_path)

    print("Saved plots to:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
