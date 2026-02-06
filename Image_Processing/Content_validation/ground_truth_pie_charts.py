#!/usr/bin/env python3
"""
Ground-truth pie chart utility.

This script generates four pie charts from one or more ground-truth CSVs
(defaults to ground_truth.csv and AURA271labels.csv if present):
1) Usability considering NFP
2) Usability considering blur (focused enough / too blurry)
3) Blur category (Blurr)
4) Quality labels (Usable / Too Blurry / Wrong Setup / Irrelevant)

Charts are saved to ~/Downloads/groundtruth_piecharts by default.

How to run:
  uv run python Image_Processing/Content_validation/ground_truth_pie_charts.py \
    --csv-paths ground_truth.csv AURA271labels.csv

  uv run python Image_Processing/Content_validation/ground_truth_pie_charts.py \
    --csv-paths ground_truth.csv AURA271labels.csv \
    --output-dir ~/Downloads/groundtruth_piecharts \
    --summary-path reports/ground_truth_piecharts_summary.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import pandas as pd


@dataclass(frozen=True)
class LabelStat:
    label: str
    count: int
    percent_of_total: float
    percent_of_labeled: float


@dataclass(frozen=True)
class ColumnStats:
    column: str
    total_rows: int
    labeled_rows: int
    missing_rows: int
    labels: tuple[LabelStat, ...]


class GroundTruthPieChartUtility:
    """Generate pie charts for selected ground-truth columns."""

    def __init__(self, csv_paths: Sequence[Path], output_dir: Path) -> None:
        if not csv_paths:
            raise ValueError("At least one CSV path is required.")
        self.csv_paths = tuple(csv_paths)
        self.output_dir = output_dir
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        missing = [path for path in self.csv_paths if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"CSV not found: {missing_text}")
        frames = [pd.read_csv(path) for path in self.csv_paths]
        self._df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return self._df

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            return self.load()
        return self._df

    def summarize_column(
        self,
        column: str,
        *,
        label_order: Optional[Sequence[str]] = None,
    ) -> ColumnStats:
        df = self._ensure_loaded()
        if column not in df.columns:
            raise ValueError(f"Column not found in CSV: {column}")

        series = df[column]
        total_rows = len(df)
        missing_rows = int(series.isna().sum())
        labeled_rows = total_rows - missing_rows
        counts = series.dropna().value_counts()

        ordered_labels = self._order_labels(counts.index.tolist(), label_order)
        label_stats: List[LabelStat] = []
        for label in ordered_labels:
            count = int(counts[label])
            label_stats.append(
                LabelStat(
                    label=label,
                    count=count,
                    percent_of_total=100.0 * count / total_rows if total_rows else 0.0,
                    percent_of_labeled=100.0 * count / labeled_rows if labeled_rows else 0.0,
                )
            )

        return ColumnStats(
            column=column,
            total_rows=total_rows,
            labeled_rows=labeled_rows,
            missing_rows=missing_rows,
            labels=tuple(label_stats),
        )

    @staticmethod
    def _order_labels(labels: Sequence[str], label_order: Optional[Sequence[str]]) -> List[str]:
        if not label_order:
            return list(labels)
        ordered = [label for label in label_order if label in labels]
        remaining = [label for label in labels if label not in label_order]
        ordered.extend(sorted(remaining))
        return ordered

    def build_pie_chart(
        self,
        column: str,
        *,
        title: str,
        output_filename: str,
        label_order: Optional[Sequence[str]] = None,
        color_map: Optional[Mapping[str, str]] = None,
        stripe_label_overrides: Optional[Mapping[str, str]] = None,
    ) -> ColumnStats:
        stats = self.summarize_column(column, label_order=label_order)
        if not stats.labels:
            raise ValueError(f"No labeled values available for column: {column}")

        values = [label.count for label in stats.labels]
        colors, stripe_colors = _resolve_wedge_colors(
            stats.labels,
            color_map=color_map,
            stripe_label_overrides=stripe_label_overrides,
        )

        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        wedges, _label_texts, pct_texts = ax.pie(
            values,
            labels=None,
            autopct=_autopct_factory(values, min_percent=0.1),
            startangle=90,
            counterclock=False,
            pctdistance=0.72,
            textprops={"fontsize": 9},
            colors=colors,
        )
        for pct_text in pct_texts:
            pct_text.set_fontsize(8)
        _apply_stripes(ax, wedges, stripe_colors)
        _place_labels_between(ax, wedges, stats.labels, radius=1.20)
        _jitter_percent_texts(
            wedges,
            stats.labels,
            pct_texts,
            label_offsets={
                "Too Blurry; Wrong Setup": (-8.0, -0.02),
                "Too Blurry, Wrong Setup": (-8.0, -0.02),
                "Irrelevant": (8.0, 0.03),
            },
        )
        ax.axis("equal")
        ax.set_title(
            f"{title}\nTotal images: {stats.total_rows}",
            fontsize=13,
            y=1.12,
        )
        fig.subplots_adjust(top=0.80)

        output_path = self.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Add padding so long labels (e.g., blurry background/foreground) stay within the PNG.
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.45)
        plt.close(fig)

        return stats

    def build_all_charts(self) -> dict[str, ColumnStats]:
        quality_colors = _build_color_map(["Usable", "Too Blurry", "Wrong Setup", "Irrelevant"])
        quality_colors["Too Blurry; Wrong Setup"] = "tab:purple"
        quality_stripe_overrides = {
            "Too Blurry; Wrong Setup": "",
            "Too Blurry, Wrong Setup": "",
        }

        return {
            "usability considering nfp": self.build_pie_chart(
                "usability considering nfp",
                title="Usability Considering NFP",
                output_filename="ground_truth_nfp_usability_pie.png",
                label_order=["usable", "not usable"],
                color_map=_build_color_map(["usable", "not usable"]),
            ),
            "usability considering blur": self.build_pie_chart(
                "usability considering blur",
                title="Usability Considering Blur",
                output_filename="ground_truth_blur_usability_pie.png",
                label_order=["focused enough", "too blurry"],
                color_map=_build_color_map(["focused enough", "too blurry"]),
            ),
            "Blurr": self.build_pie_chart(
                "Blurr",
                title="Blur Category (Blurr)",
                output_filename="ground_truth_blurr_category_pie.png",
                label_order=["blurry background", "no blurr", "blurry", "blurry foreground"],
                color_map=_build_color_map(["blurry background", "no blurr", "blurry", "blurry foreground"]),
            ),
            "Quality": self.build_pie_chart(
                "Quality",
                title="Quality Labels",
                output_filename="ground_truth_quality_pie.png",
                label_order=["Usable", "Too Blurry", "Wrong Setup", "Irrelevant"],
                color_map=quality_colors,
                stripe_label_overrides=quality_stripe_overrides,
            ),
        }

    def write_summary(self, summary_path: Path, stats_by_column: dict[str, ColumnStats]) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("# Ground-truth pie chart summary")
        lines.append("")
        source_list = ", ".join(f"`{path}`" for path in self.csv_paths)
        lines.append(f"- Source CSVs: {source_list}")
        lines.append(f"- Total images (rows): {self._ensure_loaded().shape[0]}")
        lines.append("")

        for column, stats in stats_by_column.items():
            lines.append(f"## {column}")
            lines.append(f"- Total rows: {stats.total_rows}")
            lines.append(f"- Labeled rows: {stats.labeled_rows}")
            lines.append(f"- Missing rows: {stats.missing_rows} ({_format_percent(100.0 * stats.missing_rows / stats.total_rows)})")
            lines.append("")
            lines.append("| Label | Count | % of total | % of labeled |")
            lines.append("| --- | --- | --- | --- |")
            for label in stats.labels:
                lines.append(
                    f"| {label.label} | {label.count} | {_format_percent(label.percent_of_total)} | {_format_percent(label.percent_of_labeled)} |"
                )
            lines.append("")

        lines.append("## Interpretation notes")
        nfp_note = _build_balance_note(
            stats_by_column.get("usability considering nfp"),
            positive_label="usable",
            negative_label="not usable",
            label_name="NFP usability",
            percent_kind="total",
        )
        blur_note = _build_balance_note(
            stats_by_column.get("usability considering blur"),
            positive_label="focused enough",
            negative_label="too blurry",
            label_name="blur usability",
            percent_kind="labeled",
        )
        if nfp_note and blur_note:
            lines.append(f"- There is a slight class imbalance in the primary labels. {nfp_note} {blur_note}")
        elif nfp_note:
            lines.append(f"- There is a slight class imbalance in the primary labels. {nfp_note}")
        elif blur_note:
            lines.append(f"- There is a slight class imbalance in the primary labels. {blur_note}")
        else:
            lines.append("- There is a slight class imbalance in the primary labels.")
        lines.append(
            "- The main classifier target remains **usability considering blur**. The Quality and Blurr columns explain *why* images are unusable (e.g., Wrong Setup, Irrelevant) or *how* blur manifests (foreground vs background)."
        )
        lines.append(
            "- Blurry background is still often usable; we want to use that signal to check whether the CNN correctly treats background-blur images as usable."
        )
        lines.append("")
        lines.append(
            "Note: pie chart percentages are based on labeled rows for each column; missing labels are listed above."
        )
        lines.append("")

        summary_path.write_text("\n".join(lines), encoding="utf-8")


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def _format_label_text(label: str, count: int) -> str:
    cleaned = " ".join(label.split())
    if len(cleaned) > 14:
        return f"{cleaned}\n(n={count})"
    return f"{cleaned} (n={count})"


def _split_label_parts(label: str) -> List[str]:
    parts: List[str] = []
    for chunk in label.split(";"):
        for piece in chunk.split(","):
            cleaned = piece.strip()
            if cleaned:
                parts.append(cleaned)
    return parts or [label.strip()]


def _build_color_map(label_order: Optional[Sequence[str]], *, palette_name: str = "tab10") -> dict[str, str]:
    if not label_order:
        return {}
    palette = list(plt.get_cmap(palette_name).colors)
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(label_order)
    }


def _resolve_wedge_colors(
    labels: Sequence[LabelStat],
    *,
    color_map: Optional[Mapping[str, str]],
    stripe_label_overrides: Optional[Mapping[str, str]] = None,
) -> tuple[List[str], List[Optional[str]]]:
    palette = list(plt.get_cmap("tab20").colors)
    colors: List[str] = []
    stripe_colors: List[Optional[str]] = []
    mapping = color_map or {}
    stripe_overrides = stripe_label_overrides or {}

    for index, item in enumerate(labels):
        parts = _split_label_parts(item.label)
        primary = mapping.get(item.label)
        if primary is None:
            primary = mapping.get(parts[0], palette[index % len(palette)])
        secondary = None
        if len(parts) > 1:
            override_label = stripe_overrides.get(item.label)
            if override_label == "":
                secondary = None
            elif override_label:
                secondary = mapping.get(override_label, palette[(index + 1) % len(palette)])
            else:
                secondary = mapping.get(parts[1], palette[(index + 1) % len(palette)])
        colors.append(primary)
        stripe_colors.append(secondary)

    return colors, stripe_colors


def _apply_stripes(
    ax: plt.Axes,
    wedges: Sequence[Wedge],
    stripe_colors: Sequence[Optional[str]],
    *,
    hatch: str = "///",
    outer_frac: float = 0.34,
    middle_frac: float = 0.32,
) -> None:
    for wedge, stripe_color in zip(wedges, stripe_colors):
        if not stripe_color:
            continue

        radius = wedge.r
        outer_thickness = radius * outer_frac
        middle_thickness = radius * middle_frac
        middle_outer = max(radius - outer_thickness, 0.0)
        inner_outer = max(middle_outer - middle_thickness, 0.0)

        middle = Wedge(
            center=wedge.center,
            r=middle_outer,
            theta1=wedge.theta1,
            theta2=wedge.theta2,
            width=middle_thickness,
        )
        middle.set_facecolor(wedge.get_facecolor())
        middle.set_edgecolor(stripe_color)
        middle.set_hatch(hatch)
        middle.set_linewidth(0.6)
        middle.set_zorder(wedge.zorder + 1)
        ax.add_patch(middle)

        if inner_outer > 0:
            inner = Wedge(
                center=wedge.center,
                r=inner_outer,
                theta1=wedge.theta1,
                theta2=wedge.theta2,
                width=inner_outer,
            )
            inner.set_facecolor(stripe_color)
            inner.set_edgecolor("none")
            inner.set_zorder(wedge.zorder + 1)
            ax.add_patch(inner)


def _place_labels_between(
    ax: plt.Axes,
    wedges: Sequence[Wedge],
    labels: Sequence[LabelStat],
    *,
    radius: float = 1.18,
    fontsize: int = 9,
    min_distance: float = 0.14,
) -> None:
    placed: List[tuple[float, float]] = []
    angle_offsets = (0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0, 24.0, -24.0)
    radius_offsets = (0.0, 0.06, 0.12)
    for wedge, stat in zip(wedges, labels):
        parts = _split_label_parts(stat.label)

        angle = 0.5 * (wedge.theta1 + wedge.theta2)
        if len(parts) > 1:
            angle += 0.08 * (wedge.theta2 - wedge.theta1)

        chosen_x = None
        chosen_y = None
        for r_offset in radius_offsets:
            for a_offset in angle_offsets:
                angle_rad = np.deg2rad(angle + a_offset)
                x = (radius + r_offset) * np.cos(angle_rad)
                y = (radius + r_offset) * np.sin(angle_rad)
                if all(np.hypot(x - px, y - py) >= min_distance for px, py in placed):
                    chosen_x = x
                    chosen_y = y
                    break
            if chosen_x is not None:
                break
        if chosen_x is None:
            angle_rad = np.deg2rad(angle)
            chosen_x = radius * np.cos(angle_rad)
            chosen_y = radius * np.sin(angle_rad)

        text = _format_label_text(stat.label, stat.count)
        ha = "left" if chosen_x >= 0 else "right"
        x_offset = 0.02 if chosen_x >= 0 else -0.02
        ax.text(
            chosen_x + x_offset,
            chosen_y,
            text,
            ha=ha,
            va="center",
            fontsize=fontsize,
        )
        placed.append((chosen_x + x_offset, chosen_y))


def _jitter_percent_texts(
    wedges: Sequence[Wedge],
    labels: Sequence[LabelStat],
    pct_texts: Sequence[plt.Text],
    *,
    label_offsets: Mapping[str, tuple[float, float]],
) -> None:
    for wedge, stat, pct_text in zip(wedges, labels, pct_texts):
        if not pct_text.get_text():
            continue
        offset = label_offsets.get(stat.label)
        if not offset:
            continue
        angle_offset, radius_offset = offset
        x, y = pct_text.get_position()
        radius = np.hypot(x, y)
        angle = np.arctan2(y, x) + np.deg2rad(angle_offset)
        radius = max(radius + radius_offset, 0.0)
        pct_text.set_position((radius * np.cos(angle), radius * np.sin(angle)))


def _autopct_factory(values: Sequence[int], *, min_percent: float) -> Callable[[float], str]:
    total = sum(values)
    if total <= 0:
        return lambda _pct: ""

    def _autopct(pct: float) -> str:
        if pct < min_percent:
            return ""
        if pct < 1.0:
            return f"{pct:.2f}%"
        return f"{pct:.1f}%"

    return _autopct


def _resolve_csv_paths(
    csv_paths: Optional[Sequence[Path]],
    csv_path: Optional[Path],
) -> List[Path]:
    if csv_paths:
        return list(csv_paths)
    defaults = [csv_path] if csv_path else [Path("ground_truth.csv")]
    aura_labels = Path("AURA271labels.csv")
    if aura_labels.exists() and aura_labels not in defaults:
        defaults.append(aura_labels)
    return defaults


def _find_label(stats: Optional[ColumnStats], label: str) -> Optional[LabelStat]:
    if stats is None:
        return None
    for item in stats.labels:
        if item.label == label:
            return item
    return None


def _build_balance_note(
    stats: Optional[ColumnStats],
    *,
    positive_label: str,
    negative_label: str,
    label_name: str,
    percent_kind: str,
) -> Optional[str]:
    if stats is None:
        return None

    positive = _find_label(stats, positive_label)
    negative = _find_label(stats, negative_label)
    if not positive or not negative:
        return None

    if percent_kind == "labeled":
        pos_pct = positive.percent_of_labeled
        neg_pct = negative.percent_of_labeled
        kind_text = "labeled rows"
    else:
        pos_pct = positive.percent_of_total
        neg_pct = negative.percent_of_total
        kind_text = "total rows"

    return (
        f"{label_name} skews {positive.label} "
        f"({pos_pct:.1f}% of {kind_text}) vs {negative.label} "
        f"({neg_pct:.1f}% of {kind_text})."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pie charts for ground-truth labels.")
    parser.add_argument(
        "--csv-paths",
        type=Path,
        nargs="+",
        help="One or more ground-truth CSVs. Defaults to ground_truth.csv (plus AURA271labels.csv if present).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Downloads" / "groundtruth_piecharts",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional path to write a markdown summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_paths = _resolve_csv_paths(args.csv_paths, args.csv_path)
    utility = GroundTruthPieChartUtility(csv_paths, args.output_dir)
    stats_by_column = utility.build_all_charts()
    summary_stats = dict(stats_by_column)
    summary_stats["usability considering blur"] = utility.summarize_column(
        "usability considering blur",
        label_order=["focused enough", "too blurry"],
    )

    if args.summary_path:
        utility.write_summary(args.summary_path, summary_stats)
        print(f"Summary written to {args.summary_path}")

    print(f"Pie charts saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
