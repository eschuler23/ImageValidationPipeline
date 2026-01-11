"""Copy images that exist in exactly one of two folders into a destination directory.

Images are compared via an MD5 digest of their file content. When the same
content appears in both source folders it is treated as a shared duplicate and
is excluded from the output. Files present in only one folder are copied once.
"""

from pathlib import Path
import hashlib
import shutil

# Image directories
DIR1 = Path('AURA1612')
DIR2 = Path('AURA101')
# Name output after the second directory to highlight newly discovered images.
OUTPUT_DIR = Path(f"{DIR2.name}_NEW")


def hash_image(image_path: Path) -> str:
    """Return an MD5 digest for the given image file."""
    hasher = hashlib.md5()
    with image_path.open('rb') as handle:
        hasher.update(handle.read())
    return hasher.hexdigest()


def remove_duplicates(dir1: Path, dir2: Path, output_dir: Path) -> None:
    """Copy images unique to a single directory into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_hash: dict[str, list[Path]] = {}
    dirs_by_hash: dict[str, set[Path]] = {}

    for source_dir in (dir1, dir2):
        for file_path in source_dir.iterdir():
            if not file_path.is_file():
                continue
            file_hash = hash_image(file_path)
            files_by_hash.setdefault(file_hash, []).append(file_path)
            dirs_by_hash.setdefault(file_hash, set()).add(source_dir)

    for file_hash, source_paths in files_by_hash.items():
        if len(dirs_by_hash[file_hash]) != 1:
            continue
        candidate = source_paths[0]
        target = output_dir / candidate.name
        if target.exists():
            target = output_dir / f"{candidate.stem}_{file_hash[:8]}{candidate.suffix}"
        shutil.copy(candidate, target)

    print(f"Non-duplicate images copied to {output_dir}")


remove_duplicates(DIR1, DIR2, OUTPUT_DIR)
