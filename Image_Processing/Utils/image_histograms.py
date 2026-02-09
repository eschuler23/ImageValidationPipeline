"""Generate per-image RGB histograms.

How to run (from repo root):
  uv run python Image_Processing/Utils/image_histograms.py \
    --images Imagedump271/13b594bf-3605-475a-9c39-306e6fefa209.jpeg \
             Imagedump271/512962b0-182f-4213-8601-cdb2ed441aae.jpeg \
    --output-dir ~/Downloads

  # Grayscale histogram example:
  uv run python Image_Processing/Utils/image_histograms.py \
    --images Imagedump271/13b594bf-3605-475a-9c39-306e6fefa209.jpeg \
    --output-dir ~/Downloads \
    --grayscale \
    --suffix _grayscale_histogram

  # OpenCV grayscale histogram example:
  uv run python Image_Processing/Utils/image_histograms.py \
    --images Imagedump271/13b594bf-3605-475a-9c39-306e6fefa209.jpeg \
    --output-dir ~/Downloads \
    --grayscale \
    --backend opencv \
    --suffix _opencv_grayscale_histogram
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

CHANNEL_COLORS = ("red", "green", "blue")
CHANNEL_LABELS = ("Red", "Green", "Blue")
OPENCV_CHANNEL_COLORS = ("blue", "green", "red")
OPENCV_CHANNEL_LABELS = ("Blue", "Green", "Red")
GRAYSCALE_COLOR = "black"
GRAYSCALE_LABEL = "Grayscale"
DEFAULT_BINS = 256
DEFAULT_SUFFIX = "_histogram"
DEFAULT_BACKEND = "numpy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-channel RGB histograms for one or more images."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="One or more image paths to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Downloads"),
        help="Folder to save histogram PNGs (default: ~/Downloads).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of histogram bins (default: 256).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Plot a grayscale histogram instead of per-channel RGB.",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "opencv"),
        default=DEFAULT_BACKEND,
        help="Histogram backend to use (default: numpy).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix to append to output filenames (default: _histogram).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing histogram PNGs if they already exist.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dots per inch (default: 200).",
    )
    return parser.parse_args()


def load_rgb_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    with Image.open(image_path) as image:
        # Convert to RGB so the histogram logic is consistent across image modes.
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.copy()


def load_grayscale_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    with Image.open(image_path) as image:
        if image.mode != "L":
            image = image.convert("L")
        return image.copy()


def load_opencv_image(image_path: Path, *, grayscale: bool) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV is not installed. Install cv2 or use --backend numpy.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")
    return image


def compute_histograms(image: Image.Image, bins: int) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError("Expected an RGB image with three channels.")

    histograms = []
    for channel_index in range(3):
        channel_values = image_array[:, :, channel_index].ravel()
        hist, _ = np.histogram(channel_values, bins=bins, range=(0, 256))
        histograms.append(hist)

    bin_edges = np.linspace(0, 256, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return np.stack(histograms, axis=0), bin_centers


def compute_grayscale_histogram(
    image: Image.Image, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("Expected a grayscale image with a single channel.")

    hist, bin_edges = np.histogram(image_array.ravel(), bins=bins, range=(0, 256))
    bin_edges = np.linspace(0, 256, bins + 1)
    return hist, bin_edges


def compute_opencv_histograms(
    image_bgr: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("Expected a BGR image with three channels.")
    histograms = []
    for channel_index in range(3):
        hist = cv2.calcHist([image_bgr], [channel_index], None, [bins], [0, 256])
        histograms.append(hist.reshape(-1))
    bin_edges = np.linspace(0, 256, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return np.stack(histograms, axis=0), bin_centers


def compute_opencv_grayscale_histogram(
    image_gray: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    if image_gray.ndim != 2:
        raise ValueError("Expected a grayscale image with a single channel.")
    hist = cv2.calcHist([image_gray], [0], None, [bins], [0, 256]).reshape(-1)
    bin_edges = np.linspace(0, 256, bins + 1)
    return hist, bin_edges


def plot_histogram(
    histograms: np.ndarray,
    bin_centers: np.ndarray,
    *,
    title: str,
    output_path: Path,
    dpi: int,
    channel_colors: tuple[str, str, str] = CHANNEL_COLORS,
    channel_labels: tuple[str, str, str] = CHANNEL_LABELS,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for channel_hist, color, label in zip(histograms, channel_colors, channel_labels):
        ax.plot(bin_centers, channel_hist, color=color, label=label, linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_grayscale_histogram(
    histogram: np.ndarray,
    bin_edges: np.ndarray,
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(
        bin_centers,
        histogram,
        width=bin_edges[1] - bin_edges[0],
        color=GRAYSCALE_COLOR,
        edgecolor=GRAYSCALE_COLOR,
        linewidth=0.3,
        label=GRAYSCALE_LABEL,
    )

    ax.set_title(title)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_output_path(
    output_dir: Path,
    image_path: Path,
    suffix: str,
) -> Path:
    filename = f"{image_path.stem}{suffix}.png"
    return output_dir / filename


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()

    for image_path in args.images:
        if args.backend == "opencv":
            if args.grayscale:
                image_gray = load_opencv_image(image_path, grayscale=True)
                histogram, bin_edges = compute_opencv_grayscale_histogram(
                    image_gray, bins=args.bins
                )
            else:
                image_bgr = load_opencv_image(image_path, grayscale=False)
                histograms, bin_centers = compute_opencv_histograms(
                    image_bgr, bins=args.bins
                )
        else:
            if args.grayscale:
                image = load_grayscale_image(image_path)
                histogram, bin_edges = compute_grayscale_histogram(
                    image, bins=args.bins
                )
            else:
                image = load_rgb_image(image_path)
                histograms, bin_centers = compute_histograms(image, bins=args.bins)
        output_path = build_output_path(output_dir, image_path, args.suffix)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        if args.grayscale:
            title = f"Grayscale histogram: {image_path.name}"
            plot_grayscale_histogram(
                histogram,
                bin_edges,
                title=title,
                output_path=output_path,
                dpi=args.dpi,
            )
        else:
            title = f"RGB histogram: {image_path.name}"
            channel_colors = CHANNEL_COLORS
            channel_labels = CHANNEL_LABELS
            if args.backend == "opencv":
                channel_colors = OPENCV_CHANNEL_COLORS
                channel_labels = OPENCV_CHANNEL_LABELS
            plot_histogram(
                histograms,
                bin_centers,
                title=title,
                output_path=output_path,
                dpi=args.dpi,
                channel_colors=channel_colors,
                channel_labels=channel_labels,
            )
        print(f"Saved histogram: {output_path}")


if __name__ == "__main__":
    main()
