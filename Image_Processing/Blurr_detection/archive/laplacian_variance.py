# %%
import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _scalar_to_int(value) -> int:
    return int(np.asarray(value).item())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir, os.pardir))
TRAIN_CSV_PATH = os.path.join(REPO_ROOT, 'ground_truth.csv')
GROUND_TRUTH_CSV = TRAIN_CSV_PATH
# Images are now stored under the repo-level Images folder (AURA* subfolder).
# We match by filename and skip labels that have no local image.
IMAGES_DIR = os.path.join(REPO_ROOT, 'Images')
TRAIN_IMAGE_ROOT = os.path.join(IMAGES_DIR, 'AURA1612')
PROJECT_COLUMN = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SKIP_DIR_NAMES = {'.git', '.venv', '.uv-cache', '.uv_cache', '.cache', '.mplconfig', '__pycache__'}
SORTED_USABILITY_PNG = os.path.join(
    REPO_ROOT,
    'Image_Processing',
    'Blurr_detection',
    'review_artifacts',
    'baseline_variance_sorted_by_class_cap500.png',
)
SORTED_USABILITY_PNG_CAP200 = os.path.join(
    REPO_ROOT,
    'Image_Processing',
    'Blurr_detection',
    'review_artifacts',
    'baseline_variance_sorted_by_class_cap200.png',
)

# Laplacian variance threshold.
# Prediction rule: laplacian_variance < threshold  =>  "blurry"
#                laplacian_variance >= threshold =>  "not blurry"
# To get *more* images predicted as blurry (fewer false negatives), INCREASE this value.
#LAPLACIAN_THRESHOLD = 100.0
LAPLACIAN_THRESHOLD = 186.895


def build_filename_lookup(search_root, *, extensions=IMAGE_EXTENSIONS, skip_dirs=SKIP_DIR_NAMES):
    """Index images under search_root by filename (case-insensitive)."""
    lookup = {}
    duplicates = set()

    if not search_root or not os.path.isdir(search_root):
        print(f"Warning: image search root not found: {search_root}")
        return lookup

    for root, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in skip_dirs]
        filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in extensions]
        filenames.sort()
        for filename in filenames:
            key = filename.lower()
            image_path = os.path.join(root, filename)
            if key in lookup and lookup[key] != image_path:
                duplicates.add(filename)
                continue
            lookup[key] = image_path

    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate filenames found; using the first occurrence.")

    print(f"Indexed {len(lookup)} images under: {search_root}")
    return lookup


def _pick_best_aura_dir(csv_path, images_dir):
    """Pick the AURA* folder with the most filename matches from the CSV."""
    if not images_dir or not os.path.isdir(images_dir):
        return None

    try:
        df = pd.read_csv(csv_path)
        filenames = [str(v).strip() for v in df.get('filename', []) if str(v).strip()]
    except Exception:
        return None

    if not filenames:
        return None

    candidates = [
        d for d in os.listdir(images_dir)
        if d.lower().startswith('aura') and os.path.isdir(os.path.join(images_dir, d))
    ]
    if not candidates:
        return None

    best_dir = None
    best_count = 0
    for candidate in candidates:
        path = os.path.join(images_dir, candidate)
        try:
            files = {
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            }
        except Exception:
            continue

        count = sum(1 for name in filenames if name in files)
        if count > best_count:
            best_count = count
            best_dir = path

    if best_dir and best_count > 0:
        print(f"Using image folder '{best_dir}' with {best_count} CSV matches.")
        return best_dir

    return None


def load_images_from_dir(image_dir, recursive=True):
    """Load all images from a directory (no labels).

    Returns a dict with keys: images, labels (None), filenames.
    """
    images = []
    filenames = []
    unreadable_files = 0

    if recursive:
        walker = os.walk(image_dir)
    else:
        walker = [(image_dir, [], os.listdir(image_dir))]

    for root, _, files in walker:
        for filename in files:
            if filename.startswith('.'):
                continue
            image_path = os.path.join(root, filename)
            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path)
            if image is None:
                unreadable_files += 1
                continue

            rel_path = os.path.relpath(image_path, image_dir)
            images.append(image)
            filenames.append(rel_path)

    if not images:
        raise ValueError(f'No images could be loaded from: {image_dir}')

    if unreadable_files:
        print(f"Skipped {unreadable_files} unreadable files in {image_dir}.")

    return {
        'images': images,
        'labels': None,
        'filenames': filenames,
    }


def resolve_image_path(filename, row, image_root, filename_lookup, project_column=PROJECT_COLUMN):
    """Resolve an image path by filename (optionally using project column + lookup)."""
    if not filename:
        return None

    project = (row.get(project_column) or '').strip() if project_column else ''
    if project:
        candidate = os.path.join(image_root, project, filename)
        if os.path.exists(candidate):
            return candidate

    direct = os.path.join(image_root, filename)
    if os.path.exists(direct):
        return direct

    if filename_lookup:
        key = os.path.basename(filename).lower()
        return filename_lookup.get(key)

    return None


def load_local_dataset(
    image_root,
    csv_path,
    label_column='Blurr',
    test_ratio=0.2,
    random_state=42,
    filename_lookup=None,
    project_column=PROJECT_COLUMN,
):
    """Load images described in csv_path and split them into train/test lists."""
    rng = np.random.default_rng(random_state)
    images = []
    labels = []
    filenames = []
    missing_files = 0
    unreadable_files = 0

    with open(csv_path, encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = (row.get('filename') or '').strip()
            if not filename:
                continue

            image_path = resolve_image_path(filename, row, image_root, filename_lookup, project_column)
            if not image_path or not os.path.exists(image_path):
                missing_files += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                unreadable_files += 1
                continue

            images.append(image)
            labels.append(row.get(label_column))
            filenames.append(filename)

    if not images:
        raise ValueError('No images could be loaded. Check CSV_PATH and IMAGE_DIR values.')

    if missing_files or unreadable_files:
        print(f"Skipped {missing_files} missing and {unreadable_files} unreadable files.")

    if len(images) == 1 or test_ratio <= 0.0:
        train_indices = np.arange(len(images))
        test_indices = np.array([], dtype=int)
    else:
        indices = np.arange(len(images))
        rng.shuffle(indices)
        split_index = int(len(images) * (1 - test_ratio))
        split_index = max(1, min(len(images) - 1, split_index))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

    has_labels = any(label not in (None, '') for label in labels)

    x_train = [images[i] for i in train_indices]
    x_test = [images[i] for i in test_indices]
    train_filenames = [filenames[i] for i in train_indices]
    test_filenames = [filenames[i] for i in test_indices]

    if has_labels:
        y_train = [labels[i] for i in train_indices]
        y_test = [labels[i] for i in test_indices]
    else:
        y_train = y_test = None

    return (
        {
            'images': x_train,
            'labels': y_train,
            'filenames': train_filenames,
        },
        {
            'images': x_test,
            'labels': y_test,
            'filenames': test_filenames,
        },
    )



if not os.path.isdir(TRAIN_IMAGE_ROOT):
    fallback_dir = _pick_best_aura_dir(TRAIN_CSV_PATH, IMAGES_DIR)
    if fallback_dir:
        TRAIN_IMAGE_ROOT = fallback_dir
    else:
        print(f"Warning: default image folder not found: {TRAIN_IMAGE_ROOT}")

filename_lookup = build_filename_lookup(TRAIN_IMAGE_ROOT)
train_split, _ = load_local_dataset(
    TRAIN_IMAGE_ROOT,
    TRAIN_CSV_PATH,
    test_ratio=0.0,
    filename_lookup=filename_lookup,
)
x_train = train_split['images']
y_train = train_split['labels']
train_filenames = train_split['filenames']

# Function to check if an image is blurry
def is_blurry(image, threshold=100.0):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    return laplacian_var < threshold, laplacian_var

# Initialize lists to store images
blurry_images = []
not_blurry_images = []

# Analyze the dataset
for image in x_train:
    # Check if the image is blurry
    is_blur, _ = is_blurry(image, threshold=LAPLACIAN_THRESHOLD)

    if is_blur:
        blurry_images.append(image)
    else:
        not_blurry_images.append(image)

# Output the results
total_images = len(x_train)
blurry_count = len(blurry_images)
not_blurry_count = len(not_blurry_images)

print(f"Total training images (from CSV): {total_images}")
print(f"Blurry images: {blurry_count}")
print(f"Not blurry images: {not_blurry_count}")
if total_images:
    print(f"Percentage of blurry images: {blurry_count / total_images * 100:.2f}%")

# Plot a few blurry and not blurry images to compare
def _infer_color_order(image) -> str:
    """Heuristic to guess whether a 3-channel image is BGR (OpenCV) or RGB (matplotlib).

    Returns: 'bgr' or 'rgb'
    """
    if image is None or not hasattr(image, 'shape') or len(image.shape) < 3 or image.shape[2] < 3:
        return 'bgr'
    # Typical photos tend to have higher red than blue on average. In OpenCV BGR,
    # channel 2 is red and channel 0 is blue. In RGB it's the opposite.
    b_mean = float(np.mean(image[..., 0]))
    r_mean = float(np.mean(image[..., 2]))
    if r_mean >= b_mean:
        return 'bgr'
    return 'rgb'


def _to_rgb_for_display(image, color_order: str):
    if image is None:
        return None
    if color_order == 'rgb':
        return image
    # Default: assume OpenCV BGR.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_images(images, title, num_images=5, *, color_order: str = 'bgr'):
    """Display a subset of images for a quick visual check."""
    if not images:
        return

    if color_order == 'auto':
        color_order = _infer_color_order(images[0])

    num_images = min(num_images, len(images))
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        display_img = _to_rgb_for_display(images[i], color_order)
        if display_img is not None:
            plt.imshow(display_img)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def plot_image_grid(
    images,
    titles=None,
    cols=6,
    figsize_per_col=3.0,
    figsize_per_row=3.0,
    *,
    color_order: str = 'bgr',
    title_fontsize=8,
    title_fontweight='normal',
    output_path=None,
    show=True,
):
    """Plot images in a grid with optional titles.

    color_order:
      - 'bgr': input images are OpenCV BGR (default)
      - 'rgb': input images are already RGB
      - 'auto': infer from the first image
    """
    if not images:
        return

    if color_order == 'auto':
        color_order = _infer_color_order(images[0])

    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / cols))
    fig = plt.figure(figsize=(cols * figsize_per_col, rows * figsize_per_row))

    for i, image in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        display_img = _to_rgb_for_display(image, color_order)
        if display_img is not None:
            ax.imshow(display_img)
        ax.axis('off')
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=title_fontsize, fontweight=title_fontweight)

    fig.tight_layout()
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)

# Plot blurry images
if blurry_images:
    plot_images(blurry_images, 'Blurry Images')

# Plot not blurry images
if not_blurry_images:
    plot_images(not_blurry_images, 'Not Blurry Images')


# Comparing with ground truth labels (from TRAIN_CSV_PATH)
# ground truth labels : "usability considering blur" (last column, column 5)


def normalize_ground_truth(value):
    """Reduce ground truth strings to blurry/not blurry/unknown."""
    label = (value or '').strip().lower()
    if not label:
        return 'unknown'
    if 'too blurry' in label:
        return 'blurry'
    if 'focused enough' in label:
        return 'not blurry'
    return 'unknown'


def normalize_blurr_type(value):
    """Normalize the CSV column 'Blurr' into a small set of categories."""
    label = (value or '').strip().lower()
    if not label:
        return 'unknown'
    if 'blurry foreground' in label:
        return 'blurry foreground'
    if 'blurry background' in label:
        return 'blurry background'
    # The dataset contains the misspelling "burry".
    if label == 'burry' or 'burry' in label:
        return 'burry'
    if 'no blurr' in label:
        return 'no blurr'
    return label


def parse_blurr_types_multi(value):
    """Parse possibly-multi labels from the CSV column 'Blurr'.

    Returns a list of canonical classes in:
      - 'blurry' (covers misspelling 'burry')
      - 'blurry foreground'
      - 'blurry background'
      - 'no blurr'
      - 'unknown'
    """
    label = (value or '').strip().lower()
    if not label:
        return ['unknown']

    classes = []
    if 'blurry foreground' in label:
        classes.append('blurry foreground')
    if 'blurry background' in label:
        classes.append('blurry background')
    if 'no blurr' in label:
        classes.append('no blurr')
    if label == 'burry' or 'burry' in label:
        classes.append('blurry')

    # If nothing matched but we have some non-empty label, fall back to a single normalized value.
    if not classes:
        normalized = normalize_blurr_type(value)
        if normalized == 'burry':
            normalized = 'blurry'
        classes = [normalized] if normalized else ['unknown']

    # De-dup but preserve order.
    seen = set()
    out = []
    for c in classes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def parse_usability_multi(value):
    """Parse possibly-multi labels from the 'usability considering blur' column."""
    label = (value or '').strip().lower()
    if not label:
        return ['unknown']
    classes = []
    if 'too blurry' in label:
        classes.append('too blurry')
    if 'focused enough' in label:
        classes.append('focused enough')
    if not classes:
        classes = ['unknown']
    return classes


ground_truth_labels = {}
ground_truth_blurr_types = {}
ground_truth_blurr_types_multi = {}
ground_truth_usability_multi = {}
with open(GROUND_TRUTH_CSV, encoding='utf-8', newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        filename = (row.get('filename') or '').strip()
        if not filename:
            continue
        usability_raw = row.get('usability considering blur')
        label = normalize_ground_truth(usability_raw)
        ground_truth_labels[filename] = label
        blurr_raw = row.get('Blurr')
        ground_truth_blurr_types[filename] = normalize_blurr_type(blurr_raw)
        ground_truth_blurr_types_multi[filename] = parse_blurr_types_multi(blurr_raw)
        ground_truth_usability_multi[filename] = parse_usability_multi(usability_raw)

evaluation_rows = []
for image, filename in zip(x_train, train_filenames):
    is_blur, laplacian_var = is_blurry(image, threshold=LAPLACIAN_THRESHOLD)
    predicted_label = 'blurry' if is_blur else 'not blurry'
    ground_truth_label = ground_truth_labels.get(filename, 'unknown')
    evaluation_rows.append({
        'filename': filename,
        'ground_truth': ground_truth_label,
        'prediction': predicted_label,
        'laplacian_variance': laplacian_var,
    })

df = pd.DataFrame(evaluation_rows)

variance_by_filename = {}
if not df.empty and 'filename' in df.columns and 'laplacian_variance' in df.columns:
    variance_by_filename = dict(zip(df['filename'], df['laplacian_variance']))

if df.empty:
    print('No predictions were generated.')
else:
    summary_table = pd.crosstab(
        df['ground_truth'],
        df['prediction'],
        dropna=False,
    ).reindex(index=['blurry', 'not blurry', 'unknown'], columns=['blurry', 'not blurry'], fill_value=0)

    # Keep output concise: we don't print the raw crosstab here because it largely
    # duplicates the labeled confusion matrix printed below.

    # Explicit answers for the two ground-truth labels.
    # normalize_ground_truth maps:
    # - "too blurry" -> ground_truth == "blurry"
    # - "focused enough" -> ground_truth == "not blurry"
    too_blurry_pred_blurry = _scalar_to_int(summary_table.loc['blurry', 'blurry'])
    too_blurry_pred_not_blurry = _scalar_to_int(summary_table.loc['blurry', 'not blurry'])
    focused_pred_not_blurry = _scalar_to_int(summary_table.loc['not blurry', 'not blurry'])
    focused_pred_blurry = _scalar_to_int(summary_table.loc['not blurry', 'blurry'])

    labeled_confusion = pd.DataFrame(
        {
            'Predicted blurry': [too_blurry_pred_blurry, focused_pred_blurry],
            'Predicted not blurry': [too_blurry_pred_not_blurry, focused_pred_not_blurry],
        },
        index=['Too blurry (ground truth)', 'Focused enough (ground truth)'],
    )

    print('\nConfusion matrix (labeled ground truth only):')
    print(labeled_confusion.to_string())

    unknown_row = None
    unknown_total = 0
    if 'unknown' in summary_table.index:
        unknown_row = summary_table.loc['unknown']
        unknown_total = _scalar_to_int(unknown_row.to_numpy().sum())
    print(f"\nUnknown ground truth images: {unknown_total}")
    if unknown_total and unknown_row is not None:
        unknown_pred_blurry = _scalar_to_int(unknown_row.get('blurry', 0))
        unknown_pred_not_blurry = _scalar_to_int(unknown_row.get('not blurry', 0))
        print(f"  predicted blurry: {unknown_pred_blurry}")
        print(f"  predicted not blurry: {unknown_pred_not_blurry}")

    labeled_df = df[df['ground_truth'] != 'unknown']
    print('\nDetection summary for labeled images:')
    if labeled_df.empty:
        print('No matching ground truth labels found for training images.')
    else:
        accuracy = (labeled_df['ground_truth'] == labeled_df['prediction']).mean()
        print(f"Accuracy: {accuracy:.2%} on {len(labeled_df)} labeled training images.")

        true_positives = too_blurry_pred_blurry
        true_negatives = focused_pred_not_blurry
        total_correct = true_positives + true_negatives
        print(f"Correctly classified images (TP + TN): {total_correct} of {len(labeled_df)}")
        print(f"  True positives (Too blurry predicted blurry): {true_positives}")
        print(f"  True negatives (Focused enough predicted not blurry): {true_negatives}")

        print('\nComparison using CSV column "Blurr" (foreground/burry):')
        blurr_rows = []
        for row in evaluation_rows:
            filename = row['filename']
            blurr_type = ground_truth_blurr_types.get(filename, 'unknown')
            blurr_rows.append({
                'filename': filename,
                'blurr_type': blurr_type,
                'prediction': row['prediction'],
            })

        blurr_df = pd.DataFrame(blurr_rows)
        fg_blurry_df = blurr_df[blurr_df['blurr_type'].isin(['blurry foreground', 'burry'])]
        if fg_blurry_df.empty:
            print('  No images with Blurr in {blurry foreground, burry} found.')
        else:
            predicted_blurry = int((fg_blurry_df['prediction'] == 'blurry').sum())
            predicted_not_blurry = int((fg_blurry_df['prediction'] == 'not blurry').sum())
            print(f"  Total images: {len(fg_blurry_df)}")
            print(f"  Predicted blurry: {predicted_blurry}")
            print(f"  Predicted not blurry: {predicted_not_blurry}")

            # Show false negatives for this group (ground truth says foreground/burry, model predicts not blurry).
            false_negative_filenames = fg_blurry_df.loc[
                fg_blurry_df['prediction'] == 'not blurry',
                'filename',
            ].tolist()
            if false_negative_filenames:
                filename_to_image = dict(zip(train_filenames, x_train))
                false_negative_images = [
                    filename_to_image[f]
                    for f in false_negative_filenames
                    if f in filename_to_image
                ]

                false_negative_scores = [
                    variance_by_filename.get(f)
                    for f in false_negative_filenames
                    if variance_by_filename.get(f) is not None
                ]
                if false_negative_scores:
                    fn_scores = np.asarray(false_negative_scores, dtype=float)
                    suggested_to_flip_all = float(np.max(fn_scores)) + 1e-6
                    suggested_to_flip_half = float(np.median(fn_scores))
                    print("\n  Threshold tuning hint (foreground/burry false negatives):")
                    print(f"    Current threshold: {LAPLACIAN_THRESHOLD}")
                    print(f"    FN score range: min={fn_scores.min():.3f}, median={suggested_to_flip_half:.3f}, max={fn_scores.max():.3f}")
                    print(f"    To flip ~all 16 to blurry: set threshold >= {suggested_to_flip_all:.3f} (may increase false positives)")
                    print(f"    To flip ~half of them: set threshold ~ {suggested_to_flip_half:.3f}")

                print(f"  False negatives shown below: {len(false_negative_images)}")
                # If there are many images, plot in chunks to keep figures readable.
                chunk_size = 36
                for start in range(0, len(false_negative_images), chunk_size):
                    chunk_images = false_negative_images[start:start + chunk_size]
                    chunk_titles = false_negative_filenames[start:start + chunk_size]
                    plot_image_grid(
                        chunk_images,
                        titles=chunk_titles,
                        cols=6,
                        figsize_per_col=2.5,
                        figsize_per_row=2.5,
                    )

        print('\nComparison using CSV column "Blurr" (blurry background):')
        bg_df = blurr_df[blurr_df['blurr_type'] == 'blurry background']
        if bg_df.empty:
            print('  No images with Blurr == blurry background found.')
        else:
            predicted_blurry = int((bg_df['prediction'] == 'blurry').sum())
            predicted_not_blurry = int((bg_df['prediction'] == 'not blurry').sum())
            print(f"  Total images: {len(bg_df)}")
            print(f"  Predicted blurry: {predicted_blurry}")
            print(f"  Predicted not blurry: {predicted_not_blurry}")

        # Avoid printing additional tables here; the confusion matrix above captures
        # the key breakdown already.


print('\n' + '=' * 30)
print('Done.')

# %%
# --- THRESHOLD ANALYSIS ---
# sort all images by their laplacian variance scores and visualize the distribution
# then print out all images and their scores (big red number on the image) for manual inspection sorted by score

def _overlay_score_on_image(image, score, *, color_order: str = 'bgr', font_scale=None, thickness=None, margin=20):
    """Return an image copy with a big score overlay.

    color_order controls what numeric tuple corresponds to 'red' when later displayed.
    """
    if image is None:
        return None
    out = image.copy()

    h, w = out.shape[:2]
    min_dim = min(h, w)
    if font_scale is None:
        # Scale text roughly with image size.
        font_scale = float(np.clip(min_dim / 350.0, 0.7, 3.0))
    if thickness is None:
        thickness = int(np.clip(round(font_scale * 3), 2, 10))

    text = f"{float(score):.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(np.clip(margin, 0, max(0, w - text_w - margin)))
    y = int(np.clip(margin + text_h, text_h + baseline, max(text_h + baseline, h - margin)))

    # White outline for readability.
    cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    red = (0, 0, 255) if color_order == 'bgr' else (255, 0, 0)
    cv2.putText(out, text, (x, y), font, font_scale, red, thickness, cv2.LINE_AA)
    return out


def _border_color(color_name: str, *, color_order: str):
    """Return a color tuple for named border colors."""
    name = (color_name or '').strip().lower()
    if name not in {'red', 'blue', 'yellow'}:
        name = 'red'
    if color_order == 'rgb':
        return {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
        }[name]
    # Default: BGR (OpenCV)
    return {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
    }[name]


def _add_layered_borders(image, *, color_order: str, border_px: int, border_layers):
    """Apply multiple concentric colored borders.

    border_layers is an ordered list like ['red', 'yellow', 'blue'].
    The first becomes the innermost border.
    """
    if image is None:
        return None
    if not border_layers or border_px <= 0:
        return image

    out = image
    for layer in border_layers:
        out = cv2.copyMakeBorder(
            out,
            border_px,
            border_px,
            border_px,
            border_px,
            borderType=cv2.BORDER_CONSTANT,
            value=_border_color(layer, color_order=color_order),
        )
    return out


def _plot_variance_distribution(scores, *, threshold=None, title='Laplacian variance distribution'):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    # If the distribution is extremely skewed, a few huge outliers can make the
    # histogram look like “everything is 0”. We cap the visible range to p99.
    p99 = float(np.quantile(scores, 0.99)) if scores.size > 10 else float(np.max(scores))
    max_visible = p99 if p99 > 0 else float(np.max(scores))
    if max_visible <= 0:
        max_visible = 1.0

    # If values are highly clustered near zero, a non-linear x-scale helps.
    positive = scores[(scores > 0) & (scores <= max_visible)]
    use_symlog_x = False
    if positive.size >= 2:
        ratio = float(np.max(positive) / np.min(positive))
        use_symlog_x = ratio >= 1e3

    if use_symlog_x and positive.size >= 2:
        min_pos = float(np.min(positive))
        min_edge = max(min_pos, 1e-6)
        if max_visible > min_edge:
            log_bins = np.logspace(np.log10(min_edge), np.log10(max_visible), num=60)
            bins = np.unique(np.concatenate(([0.0], log_bins)))
            ax.hist(scores, bins=bins, color='#4c78a8', alpha=0.85)
            ax.set_xscale('symlog', linthresh=min_edge * 10)
        else:
            ax.hist(scores, bins=60, range=(0.0, max_visible), color='#4c78a8', alpha=0.85)
    else:
        ax.hist(scores, bins=60, range=(0.0, max_visible), color='#4c78a8', alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel('Laplacian variance')
    ax.set_ylabel('Count')
    if threshold is not None:
        ax.axvline(float(threshold), color='red', linestyle='--', linewidth=2, label=f'Threshold = {float(threshold):.3f}')
        ax.legend(loc='upper right')
    ax.set_xlim(0.0, max_visible)

    if scores.size > 0:
        ax.text(
            0.98,
            0.98,
            f"min={scores.min():.3f}\nmedian={np.median(scores):.3f}\nmax={scores.max():.3f}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'),
        )

    ax = axes[1]
    sorted_scores = np.sort(scores)
    ranks = np.arange(sorted_scores.size)
    ax.scatter(ranks, sorted_scores, s=8, alpha=0.65, color='#72b7b2', edgecolors='none')
    ax.set_title('Sorted scores (ascending, per image)')
    ax.set_xlabel('Rank (0 = lowest score)')
    ax.set_ylabel('Laplacian variance')
    if threshold is not None:
        ax.axhline(float(threshold), color='red', linestyle='--', linewidth=2)

    # If there’s a huge dynamic range, use log scale so low scores aren’t flattened.
    positive = sorted_scores[sorted_scores > 0]
    if positive.size >= 2:
        ratio = float(np.max(positive) / np.min(positive))
        if ratio >= 1e3:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


def _plot_sorted_scores_by_class(
    analysis_df,
    *,
    score_col='laplacian_variance',
    classes_col=None,
    title='All images sorted by Laplacian variance',
    threshold=None,
    palette=None,
    class_order=None,
    y_cap_quantile=0.99,
    y_cap_value=None,
    output_path=None,
    show_plot=True,
):
    """Bar plot: images sorted by score; bars colored by ground-truth class.

    Supports multi-class per image by rendering multiple thin bars with small x-offsets.
    """
    if analysis_df is None or analysis_df.empty:
        return
    if score_col not in analysis_df.columns:
        return

    df_local = analysis_df.copy()
    df_local[score_col] = pd.to_numeric(df_local[score_col], errors='coerce')
    df_local = df_local.dropna(subset=[score_col])
    if df_local.empty:
        return

    df_local = df_local.sort_values(score_col, ascending=True).reset_index(drop=True)
    df_local['rank'] = np.arange(len(df_local))

    if classes_col is None or classes_col not in df_local.columns:
        df_local['_classes'] = [['unknown'] for _ in range(len(df_local))]
        classes_col = '_classes'

    def _ensure_list(v):
        if isinstance(v, (list, tuple)):
            return list(v) if v else ['unknown']
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ['unknown']
        return [str(v)]

    class_lists = df_local[classes_col].apply(_ensure_list)

    if palette is None:
        palette = {'unknown': '#9d9d9d'}

    scores = df_local[score_col].to_numpy(dtype=float)
    cap = float(np.quantile(scores, y_cap_quantile)) if scores.size > 10 else float(np.max(scores))
    if not np.isfinite(cap) or cap <= 0:
        cap = float(np.max(scores)) if np.max(scores) > 0 else 1.0
    if y_cap_value is not None:
        cap = float(y_cap_value)

    # Build per-class x/y arrays (with offsets for multi-class rows).
    per_class = {}
    outliers = 0
    for rank, score, classes in zip(df_local['rank'].to_numpy(), scores, class_lists.tolist()):
        n = len(classes)
        offsets = [0.0] if n <= 1 else np.linspace(-0.25, 0.25, n)
        y = float(score)
        if y > cap:
            outliers += 1
            y = cap
        for off, cls in zip(offsets, classes):
            cls = 'unknown' if not cls else str(cls)
            per_class.setdefault(cls, {'x': [], 'y': []})
            per_class[cls]['x'].append(rank + float(off))
            per_class[cls]['y'].append(y)

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Render classes in stable order.
    keys = list(per_class.keys())
    if class_order:
        keys = [k for k in class_order if k in per_class] + [k for k in keys if k not in set(class_order)]

    import matplotlib.patches as mpatches

    handles = []
    for cls in keys:
        color = palette.get(cls, '#9d9d9d')
        x = np.asarray(per_class[cls]['x'], dtype=float)
        y = np.asarray(per_class[cls]['y'], dtype=float)
        ax.vlines(x, 0, y, colors=color, linewidth=1.0, alpha=0.95)
        handles.append(mpatches.Patch(color=color, label=cls))

    if threshold is not None:
        ax.axhline(float(threshold), color='red', linestyle='--', linewidth=2, label='threshold')

    ax.set_title(title)
    ax.set_xlabel('Rank (0 = lowest score)')
    ax.set_ylabel('Laplacian variance')
    ax.set_ylim(0, cap * 1.02)

    if outliers:
        ax.text(
            0.01,
            0.98,
            (
                f"Clipped {outliers} / {len(df_local)} values at "
                + (f"max={cap:.0f}" if y_cap_value is not None else f"p{int(y_cap_quantile*100)}={cap:.1f}")
                + " for visibility"
            ),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'),
        )

    ax.legend(handles=handles, loc='upper right', fontsize=8)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_sorted_scores_by_usability(
    analysis_df,
    *,
    threshold=None,
    output_path=None,
    show_plot=True,
    y_cap_value=500,
):
    palette = {
        'focused enough': '#54a24b',
        'too blurry': '#e45756',
        'unknown': '#9d9d9d',
    }
    _plot_sorted_scores_by_class(
        analysis_df,
        classes_col='usability_multi',
        title='Images sorted by Laplacian variance (colored by usability considering blur)',
        threshold=threshold,
        palette=palette,
        class_order=['focused enough', 'too blurry', 'unknown'],
        y_cap_quantile=0.99,
        y_cap_value=y_cap_value,
        output_path=output_path,
        show_plot=show_plot,
    )


def _plot_sorted_scores_by_blurr_type(analysis_df, *, threshold=None):
    palette = {
        'blurry': '#f58518',
        'blurry foreground': '#eeca3b',
        'blurry background': '#4c78a8',
        'no blurr': '#54a24b',
        'unknown': '#9d9d9d',
    }
    _plot_sorted_scores_by_class(
        analysis_df,
        classes_col='blurr_types_multi',
        title='Images sorted by Laplacian variance (colored by blur type ground truth)',
        threshold=threshold,
        palette=palette,
        class_order=['blurry', 'blurry foreground', 'blurry background', 'no blurr', 'unknown'],
        y_cap_quantile=0.99,
        y_cap_value=500,
    )


def _show_border_legend():
    """Show a legend explaining what each colored border means."""
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color='red', label='Red border: usability considering blur = "too blurry"'),
        mpatches.Patch(color='yellow', label='Yellow border: blur type = "blurry foreground" or "blurry" (burry)'),
        mpatches.Patch(color='blue', label='Blue border: blur type = "blurry background"'),
    ]

    fig = plt.figure(figsize=(12, 1.4))
    fig.legend(handles=handles, loc='center', ncol=1, frameon=True, title='Borders can be layered (concentric)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_scored_images_sorted(
    images_bgr,
    filenames,
    scores,
    *,
    cols=6,
    chunk_size=36,
    max_images=None,
    save_dir=None,
    color_order: str = 'auto',
    highlight_filenames=None,
    border_layers_by_filename=None,
    include_filename_in_title=True,
    score_title_fontsize=8,
    score_title_fontweight='normal',
    output_grid_dir=None,
    only_chunk_index=None,
    show_grids=True,
):
    """Display images sorted by score with a big red score overlay.

    - chunks the display to keep figures responsive
    - optionally saves annotated images to save_dir (preserving relative paths)
    - optionally saves grid figures to output_grid_dir
    """
    if not images_bgr:
        return

    cols = max(1, int(cols))
    chunk_size = max(cols, int(chunk_size))
    if only_chunk_index is not None:
        only_chunk_index = int(only_chunk_index)
        if only_chunk_index <= 0:
            raise ValueError('only_chunk_index must be >= 1 when provided.')

    if max_images is not None:
        max_images = int(max_images)

    items = list(zip(images_bgr, filenames, scores))
    if max_images is not None:
        items = items[:max_images]

    # Infer channel order once for the batch; this avoids the "blue skin" issue when
    # images are already RGB in memory.
    def _infer_color_order_fallback(image) -> str:
        if image is None or not hasattr(image, 'shape') or len(image.shape) < 3 or image.shape[2] < 3:
            return 'bgr'
        b_mean = float(np.mean(image[..., 0]))
        r_mean = float(np.mean(image[..., 2]))
        return 'bgr' if r_mean >= b_mean else 'rgb'

    if color_order not in {'auto', 'bgr', 'rgb'}:
        raise ValueError("color_order must be one of {'auto','bgr','rgb'}")

    inferred_order: str
    if color_order == 'auto':
        infer_fn = globals().get('_infer_color_order')
        if callable(infer_fn):
            inferred_order = str(infer_fn(items[0][0])) if items else 'bgr'
        else:
            inferred_order = _infer_color_order_fallback(items[0][0]) if items else 'bgr'
    else:
        inferred_order = color_order

    if inferred_order not in {'bgr', 'rgb'}:
        inferred_order = 'bgr'

    highlight_set = set(highlight_filenames or [])
    border_layers_by_filename = border_layers_by_filename or {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if output_grid_dir:
        os.makedirs(output_grid_dir, exist_ok=True)

    # Compatibility: in notebooks it’s easy to have an older plot_image_grid() definition
    # loaded that doesn’t accept color_order. We adapt at runtime.
    try:
        import inspect

        _plot_grid_params = set(inspect.signature(plot_image_grid).parameters.keys())
        _plot_grid_supports_color_order = 'color_order' in _plot_grid_params
    except Exception:
        _plot_grid_params = set()
        _plot_grid_supports_color_order = False

    for start in range(0, len(items), chunk_size):
        chunk_index = (start // chunk_size) + 1
        if only_chunk_index is not None and chunk_index != only_chunk_index:
            continue

        chunk = items[start:start + chunk_size]
        chunk_images = []
        chunk_titles = []

        for image_bgr, filename, score in chunk:
            scored = _overlay_score_on_image(image_bgr, score, color_order=inferred_order)
            if scored is None:
                continue

            h, w = scored.shape[:2]
            base_border_px = int(np.clip(round(min(h, w) * 0.02), 3, 36))
            border_px = int(np.clip(base_border_px * 3, 6, 72))

            # Layer borders so multiple classes remain visible.
            layers = []
            if filename in highlight_set:
                layers.append('red')

            extra_layers = border_layers_by_filename.get(filename) or []
            for layer in extra_layers:
                if layer not in layers:
                    layers.append(layer)

            scored = _add_layered_borders(
                scored,
                color_order=inferred_order,
                border_px=border_px,
                border_layers=layers,
            )

            if save_dir:
                out_path = os.path.join(save_dir, filename)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                # cv2.imwrite expects BGR ordering.
                to_write = scored
                if to_write is not None and inferred_order == 'rgb':
                    to_write = cv2.cvtColor(np.asarray(to_write), cv2.COLOR_RGB2BGR)
                if to_write is not None:
                    cv2.imwrite(out_path, np.asarray(to_write))

            chunk_images.append(scored)
            score_text = f"{float(score):.1f}"
            if include_filename_in_title:
                chunk_titles.append(f"{filename}\n{score_text}")
            else:
                chunk_titles.append(score_text)

        if not chunk_images:
            continue

        grid_output_path = None
        if output_grid_dir:
            grid_output_path = os.path.join(
                output_grid_dir,
                f"sorted_laplacian_grid_{chunk_index:02d}.png",
            )

        common_kwargs = {
            'titles': chunk_titles,
            'cols': cols,
            'figsize_per_col': 2.6,
            'figsize_per_row': 2.6,
        }
        if 'title_fontsize' in _plot_grid_params:
            common_kwargs['title_fontsize'] = score_title_fontsize
        if 'title_fontweight' in _plot_grid_params:
            common_kwargs['title_fontweight'] = score_title_fontweight
        if grid_output_path and 'output_path' in _plot_grid_params:
            common_kwargs['output_path'] = grid_output_path
        if 'show' in _plot_grid_params:
            common_kwargs['show'] = show_grids

        if _plot_grid_supports_color_order:
            common_kwargs['color_order'] = inferred_order
            plot_image_grid(chunk_images, **common_kwargs)
        else:
            # Older plot_image_grid assumes BGR input and converts BGR->RGB internally.
            to_plot = chunk_images
            if inferred_order == 'rgb':
                to_plot = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in chunk_images]
            plot_image_grid(to_plot, **common_kwargs)

        if grid_output_path:
            print(f"Saved sorted Laplacian grid chunk {chunk_index:02d}: {grid_output_path}")


if df.empty:
    print('Threshold analysis skipped: no predictions/scores available.')
else:
    # Build a stable mapping from filename -> image so we can sort + display.
    filename_to_image = dict(zip(train_filenames, x_train))
    analysis_df = df.dropna(subset=['laplacian_variance']).copy()

    # Notebook safety: users often run only parts of the file, so these maps may not
    # exist in the kernel. Build fallbacks from the CSV if needed.
    gt_blurr_multi = globals().get('ground_truth_blurr_types_multi')
    gt_usability_multi = globals().get('ground_truth_usability_multi')
    if not isinstance(gt_blurr_multi, dict) or not isinstance(gt_usability_multi, dict):
        gt_blurr_multi = {}
        gt_usability_multi = {}

        def _parse_blurr_multi_fallback(value):
            s = (value or '').strip().lower()
            if not s:
                return ['unknown']
            classes = []
            if 'blurry foreground' in s:
                classes.append('blurry foreground')
            if 'blurry background' in s:
                classes.append('blurry background')
            if 'no blurr' in s:
                classes.append('no blurr')
            if s == 'burry' or 'burry' in s:
                classes.append('blurry')
            if not classes:
                classes = ['unknown']
            # De-dup preserving order.
            out = []
            seen = set()
            for c in classes:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        def _parse_usability_multi_fallback(value):
            s = (value or '').strip().lower()
            if not s:
                return ['unknown']
            classes = []
            if 'too blurry' in s:
                classes.append('too blurry')
            if 'focused enough' in s:
                classes.append('focused enough')
            return classes or ['unknown']

        try:
            with open(GROUND_TRUTH_CSV, encoding='utf-8', newline='') as _csv_file:
                _reader = csv.DictReader(_csv_file)
                for _row in _reader:
                    _fn = (_row.get('filename') or '').strip()
                    if not _fn:
                        continue
                    gt_blurr_multi[_fn] = _parse_blurr_multi_fallback(_row.get('Blurr'))
                    gt_usability_multi[_fn] = _parse_usability_multi_fallback(_row.get('usability considering blur'))
        except Exception as e:
            print(f"Warning: could not rebuild multi-label ground truth maps: {e}")
            gt_blurr_multi = {}
            gt_usability_multi = {}

    # Attach ground-truth blur class from the CSV column "Blurr".
    analysis_df['blurr_type'] = analysis_df['filename'].map(ground_truth_blurr_types).fillna('unknown')

    # Multi-label versions (lists) for plotting + multi-class handling.
    analysis_df['blurr_types_multi'] = analysis_df['filename'].map(gt_blurr_multi)
    analysis_df['blurr_types_multi'] = analysis_df['blurr_types_multi'].apply(
        lambda v: v if isinstance(v, list) and v else ['unknown']
    )
    analysis_df['usability_multi'] = analysis_df['filename'].map(gt_usability_multi)
    analysis_df['usability_multi'] = analysis_df['usability_multi'].apply(
        lambda v: v if isinstance(v, list) and v else ['unknown']
    )

    # Sort by score (ascending: blurriest first by this metric).
    analysis_df = analysis_df.sort_values('laplacian_variance', ascending=True).reset_index(drop=True)

    scores = analysis_df['laplacian_variance'].to_numpy(dtype=float)
    _plot_variance_distribution(scores, threshold=LAPLACIAN_THRESHOLD)
    _plot_sorted_scores_by_usability(
        analysis_df,
        threshold=None,
        output_path=SORTED_USABILITY_PNG,
        y_cap_value=500,
    )
    _plot_sorted_scores_by_usability(
        analysis_df,
        threshold=None,
        output_path=SORTED_USABILITY_PNG_CAP200,
        y_cap_value=200,
    )
    _plot_sorted_scores_by_blurr_type(analysis_df, threshold=None)
    _show_border_legend()

    sorted_images = []
    sorted_filenames = []
    sorted_scores = []
    blurry_ground_truth = set(
        analysis_df.loc[analysis_df['ground_truth'] == 'blurry', 'filename'].tolist()
    )

    # Additional border layers requested:
    # - blue: blurry background
    # - yellow: blurry foreground (and dataset misspelling 'burry')
    # Use the multi-label column so we handle images with more than one class.
    fg_ground_truth = set(
        analysis_df.loc[
            analysis_df['blurr_types_multi'].apply(lambda xs: isinstance(xs, list) and ('blurry foreground' in xs or 'blurry' in xs)),
            'filename',
        ].tolist()
    )
    bg_ground_truth = set(
        analysis_df.loc[
            analysis_df['blurr_types_multi'].apply(lambda xs: isinstance(xs, list) and ('blurry background' in xs)),
            'filename',
        ].tolist()
    )
    border_layers_by_filename = {}
    for fn in fg_ground_truth:
        border_layers_by_filename.setdefault(fn, []).append('yellow')
    for fn in bg_ground_truth:
        border_layers_by_filename.setdefault(fn, []).append('blue')
    missing = 0
    for _, row in analysis_df.iterrows():
        filename = row['filename']
        image = filename_to_image.get(filename)
        if image is None:
            missing += 1
            continue
        sorted_images.append(image)
        sorted_filenames.append(filename)
        sorted_scores.append(float(row['laplacian_variance']))

    if missing:
        print(f"Threshold analysis: skipped {missing} images missing from in-memory dataset.")

    print(f"\nShowing {len(sorted_images)} images sorted by Laplacian variance (ascending).")
    print("Tip: set max_images=200 if this is too slow.")

    # Optional: save annotated images for offline inspection.
    SAVE_SCORED_IMAGES = False
    SCORED_OUTPUT_DIR = os.path.join(BASE_DIR, 'threshold_analysis_scored')
    GRID_OUTPUT_DIR = os.path.join(REPO_ROOT, 'Image_Processing', 'Blurr_detection', 'review_artifacts', 'sorted_laplacian_grids')
    GRID_CHUNK_INDEX_RAW = (os.getenv('LAPLACIAN_GRID_CHUNK_INDEX') or '').strip()
    GRID_SHOW = (os.getenv('LAPLACIAN_GRID_SHOW') or '').strip().lower() in {'1', 'true', 'yes', 'y'}

    GRID_CHUNK_INDEX = None
    if GRID_CHUNK_INDEX_RAW:
        try:
            GRID_CHUNK_INDEX = int(GRID_CHUNK_INDEX_RAW)
        except ValueError:
            print(
                'Warning: LAPLACIAN_GRID_CHUNK_INDEX must be an integer (1-based). '
                'Rendering all chunks instead.'
            )
            GRID_CHUNK_INDEX = None

    show_scored_images_sorted(
        sorted_images,
        sorted_filenames,
        sorted_scores,
        cols=6,
        chunk_size=36,
        max_images=None,
        save_dir=SCORED_OUTPUT_DIR if SAVE_SCORED_IMAGES else None,
        highlight_filenames=blurry_ground_truth,
        border_layers_by_filename=border_layers_by_filename,
        include_filename_in_title=False,
        score_title_fontsize=24,
        score_title_fontweight='bold',
        output_grid_dir=GRID_OUTPUT_DIR,
        only_chunk_index=GRID_CHUNK_INDEX,
        show_grids=GRID_SHOW,
    )
# %%
