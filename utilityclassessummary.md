# Utility Classes Summary

This file maps preprocessing-related utility classes to the tasks you listed and describes each class in a single sentence with its role.

## 1) Labeln
- `Sample` (`Image_Processing/Content_validation/train_binary_classifier.py`): Stores per-image label and raw label plus filename/project metadata used throughout preprocessing and training.
- `SampleVariant` (`Image_Processing/Content_validation/train_binary_classifier.py`): Tracks augmented variants with label, label source, and optional blur radius so label changes remain explainable.

## 2) Daten Laden / Validierungsdatensatz laden
- `CsvImageDataset` (`Image_Processing/Content_validation/train_binary_classifier.py`): Torch dataset that loads images and labels from `SampleVariant` lists for train/val/test splits.
- `FolderImageDataset` (`Image_Processing/Content_validation/predict_folder.py`): Dataset that loads images from a folder for inference-style validation runs.
- `FolderDataset` (`Image_Processing/Content_validation/misclassification_grid.py`): Dataset that loads image tensors and filenames for visual grid evaluation.

## 3) Labels Laden
- `GroundTruthRow` (`Image_Processing/Content_validation/pipeline_steps/loading.py`): Represents a CSV row with `label_raw` and untouched filename/project fields for the preprocessing pipeline.
- `GroundTruthRow` (`Image_Processing/Content_validation/ground_truth_analysis.py`): Stores a raw CSV row plus normalized filename for label-distribution analysis.
- `CsvEntry` (`Image_Processing/Content_validation/check_ground_truth.py`): Captures CSV label and filename fields to support label/file consistency checks.

## 4) Bilder von Server auf die lokale Maschine kopieren und doppelte Bilder entfernen
- No class in this repo: `Image_Processing/Utils/remove_duplicate_files.py` implements this as module-level functions (for example `copy_unique_images`) that hash and copy unique files.

## 5) Nochmals Doppelte Bilder entfernen
- `GroundTruthAnalyzer` (`Image_Processing/Content_validation/ground_truth_analysis.py`): Provides optional filename deduping (`dedupe_filenames`) when reloading CSVs for analysis.

## 6) Pruefen ob jedes Bild ein zugehoeriges Label hat
- `CsvEntry` (`Image_Processing/Content_validation/check_ground_truth.py`): Represents each CSV row used to report missing labels or missing files during ground-truth checks.

## 7) Mit Leerzeichen im Dateinamen umgehen
- `CsvEntry` (`Image_Processing/Content_validation/check_ground_truth.py`): Keeps both raw and normalized filenames so spaces and special characters are preserved while matching on disk.
- `GroundTruthRow` (`Image_Processing/Content_validation/pipeline_steps/loading.py`): Stores filenames as-is and optionally decodes percent-newlines for safer matching.

## 8) Bilder Komprimieren
- `RandomJpegCompression` (`Image_Processing/Content_validation/train_binary_classifier.py`): Applies randomized JPEG compression as a preprocessing/augmentation step.

## 9) Diverse Hilfsklassen fuer die visuelle Ueberpruefung
- `Misclassification` (`Image_Processing/Content_validation/misclassification_grid.py`): Bundles per-image true/pred labels and metadata for rendering misclassification grids.
- `FolderDataset` (`Image_Processing/Content_validation/misclassification_grid.py`): Loads image tensors and filenames to feed grid visualizations.
- `Sample` (`Image_Processing/Content_validation/blur_threshold_calibration.py`): Holds per-image label and path info for blur-threshold review grids.
- `Sample` (`Image_Processing/Content_validation/size_category_review.py`): Captures size metadata for selecting examples in size-category review grids.
- `Sample` (`Image_Processing/Content_validation/size_aware_blur_review.py`): Captures size metadata for blur-augmented review grids.
- `GroundTruthPieChartUtility` (`Image_Processing/Content_validation/ground_truth_pie_charts.py`): Generates label-distribution pie charts for visual inspection.
- `LabelStat` (`Image_Processing/Content_validation/ground_truth_pie_charts.py`): Represents a single label's counts and percentages for charting.
- `ColumnStats` (`Image_Processing/Content_validation/ground_truth_pie_charts.py`): Summarizes per-column label counts used by chart rendering.
- `ConfusionStats` (`Image_Processing/Content_validation/aura271_model_comparison_chart.py`): Holds TP/FP/FN/TN counts for comparison charts.
- `RunRecord` (`Image_Processing/Content_validation/run_summary_charts.py`): Stores per-run metrics used to plot training summary charts.

## 10) Hilfsklasse Mapping bei veraenderten Namen der Ordner
- `CsvEntry` (`Image_Processing/Content_validation/check_ground_truth.py`): Includes `project_mapped` to apply project/folder name mappings when checking labels.
- `Sample` (`Image_Processing/Content_validation/train_binary_classifier.py`): Includes `project_mapped` so image paths resolve correctly after applying the project map.
