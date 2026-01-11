# Blur detection agent

## Context
- Prior attempt in the archive: Laplacian variance on full images.
- Result: no clean threshold separated blurry vs non-blurry images.
- Goal now: explore other approaches and record options.

## Brainstormed directions

### 1) Foreground-only Laplacian variance
- Use a promptable segmentation model (e.g., Segment Anything) to mask foreground objects.
- Compute Laplacian variance only on foreground pixels, not the background.
- Idea: reduce background noise so blur signal is stronger for the objects of interest.
- Open questions: how to handle multiple objects, small masks, or failures in segmentation.

### 2) Alternative classical blur metrics
- Tenengrad/Sobel gradient energy
- Brenner gradient
- FFT-based high-frequency energy ratio
- Wavelet-based sharpness measures
- Edge density or Canny edge statistics
- Variance of local contrast or local entropy

### 3) Multi-feature classifier
- Extract a small set of blur-related features (from section 2).
- Train a simple classifier (logistic regression, SVM, random forest).
- Use ground truth labels to learn decision boundaries instead of a single threshold.

### 4) CNN-based binary classifier
- Fine-tune a small CNN (e.g., MobileNet/ResNet18) on labeled blur/non-blur data.
- Use transfer learning to reduce training time and data needs.
- Optionally train on patches or multi-crops for more robust predictions.

### 5) Hybrid approach
- Use segmentation + classical metrics as features for a lightweight classifier.
- Or use a CNN but restrict evaluation to foreground crops.

## Next steps
- Define evaluation split with the ground truth CSV.
- Choose a first approach to prototype and compare against the baseline Laplacian variance.
