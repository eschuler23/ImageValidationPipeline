"""Copy new images from a source folder that are not in a reference folder.

Images are compared via an MD5 digest of their file content.

How to run (from repo root):
    uv run python Image_Processing/Utils/remove_duplicate_files.py \\
        --source Imagedump271 \\
        --reference Images \\
        --output Images/AURA271 \\
        --recursive
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable

HASH_CHUNK_SIZE = 1024 * 1024
DEFAULT_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images from a source folder that do not appear in a reference folder."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Imagedump271"),
        help="Folder containing newly dumped images.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("Images"),
        help="Folder containing existing images to compare against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Images/AURA271"),
        help="Folder to receive the unique images from the source.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan source and reference folders recursively.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level of each folder.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(ext.lstrip(".") for ext in DEFAULT_EXTENSIONS)),
        help=(
            "Comma-separated list of extensions to treat as images (default: common image types)."
        ),
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Ignore extension filtering and hash all files.",
    )
    return parser.parse_args()


def normalize_extensions(extensions_csv: str) -> set[str]:
    extensions: set[str] = set()
    for raw_value in extensions_csv.split(","):
        value = raw_value.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        extensions.add(value)
    return extensions


def iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*")
    else:
        yield from root.iterdir()


def iter_image_files(
    root: Path,
    recursive: bool,
    extensions: set[str] | None,
) -> Iterable[Path]:
    for file_path in iter_files(root, recursive=recursive):
        if not file_path.is_file():
            continue
        if extensions is not None and file_path.suffix.lower() not in extensions:
            continue
        yield file_path


def hash_file(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_reference_hashes(
    reference_dir: Path,
    recursive: bool,
    extensions: set[str] | None,
) -> tuple[set[str], int]:
    hashes: set[str] = set()
    count = 0
    for file_path in iter_image_files(reference_dir, recursive=recursive, extensions=extensions):
        hashes.add(hash_file(file_path))
        count += 1
    return hashes, count


def resolve_target_path(
    output_dir: Path,
    source_path: Path,
    file_hash: str,
) -> Path:
    target = output_dir / source_path.name
    if target.exists():
        target = output_dir / f"{source_path.stem}_{file_hash[:8]}{source_path.suffix}"
    return target


def copy_unique_images(
    source_dir: Path,
    reference_hashes: set[str],
    output_dir: Path,
    recursive: bool,
    extensions: set[str] | None,
) -> tuple[int, int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_existing = 0
    skipped_source_dupe = 0
    source_count = 0
    seen_source_hashes: set[str] = set()

    for source_path in iter_image_files(source_dir, recursive=recursive, extensions=extensions):
        source_count += 1
        file_hash = hash_file(source_path)
        if file_hash in reference_hashes:
            skipped_existing += 1
            continue
        if file_hash in seen_source_hashes:
            skipped_source_dupe += 1
            continue
        seen_source_hashes.add(file_hash)
        target = resolve_target_path(output_dir, source_path, file_hash)
        shutil.copy2(source_path, target)
        copied += 1

    return copied, skipped_existing, skipped_source_dupe, source_count


def validate_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} folder does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"{label} path is not a directory: {path}")


def main() -> None:
    args = parse_args()

    validate_directory(args.source, "Source")
    validate_directory(args.reference, "Reference")

    extensions = None if args.all_files else normalize_extensions(args.extensions)
    reference_hashes, reference_count = collect_reference_hashes(
        args.reference,
        recursive=args.recursive,
        extensions=extensions,
    )
    copied, skipped_existing, skipped_source_dupe, source_count = copy_unique_images(
        args.source,
        reference_hashes,
        args.output,
        recursive=args.recursive,
        extensions=extensions,
    )

    print("Unique image copy complete.")
    print(f"Reference images scanned: {reference_count}")
    print(f"Source images scanned: {source_count}")
    print(f"Copied to {args.output}: {copied}")
    print(f"Skipped (already in reference): {skipped_existing}")
    print(f"Skipped (duplicate within source): {skipped_source_dupe}")


if __name__ == "__main__":
    main()
