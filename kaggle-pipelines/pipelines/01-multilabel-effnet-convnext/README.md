# README.md

# BirdCLEF 2026 — Pipeline 1
# Strong Multi-Label Soundscape Baseline

> Note:- Kaggle notebook only, so we need a cell at the top for configs, to set the paths on notebook

---

# Overview

This pipeline is the recommended first production-grade baseline for BirdCLEF 2026.

It is designed specifically around the EDA findings:
- strong multi-label structure,
- severe class imbalance,
- overlapping acoustic events,
- noisy field recordings,
- heterogeneous taxa.

The pipeline focuses on:
- stability,
- fast iteration,
- leaderboard competitiveness,
- scalability into ensembles and stacking later.

It trains on the labeled clip set in `train.csv` plus the labeled soundscape segments in `train_soundscapes_labels.csv`, and writes submissions in the full official schema from `sample_submission.csv`.

---

# Core Idea

Instead of treating BirdCLEF as:
- single-label classification,
- or entire-audio prediction,

we treat it as:

## Multi-label soundscape classification using sliding windows.

Each audio recording is:
1. split into short windows,
2. converted into mel spectrograms,
3. processed independently,
4. aggregated into final predictions.

---

# Pipeline Architecture

```text
Audio (.ogg)
      ↓
Sliding Window Segmentation
(5-second windows)
      ↓
Waveform Preprocessing
      ↓
Log-Mel Spectrogram
      ↓
CNN Backbone
(EfficientNet / ConvNeXt)
      ↓
Sigmoid Multi-label Head
      ↓
Window Prediction Aggregation
      ↓
Per-Class Threshold Calibration
      ↓
Final Submission
```

---

# Why This Pipeline Fits BirdCLEF

The EDA showed:

| Observation | Pipeline Response |
|---|---|
| Multi-label soundscapes | Sigmoid multi-label head |
| Overlapping species | Sliding-window inference |
| Long-tail imbalance | Weighted BCE / focal loss |
| Acoustic heterogeneity | Spectrogram-based learning |
| Noisy environments | Augmentation + TTA |
| Temporal variability | Overlapping temporal windows |

---

# Recommended Audio Configuration

## Sample Rate
```python
SR = 32000
```

Reason:
- preserves bird harmonics,
- widely used in BirdCLEF solutions,
- good compute/performance tradeoff.

---

## Window Length
```python
WINDOW_SECONDS = 5
```

Reason:
- aligns with soundscape annotations,
- enough context for bird phrases,
- computationally efficient.

---

## Hop Length (Inference)
```python
HOP_SECONDS = 2.5
```

Reason:
- overlapping windows improve recall,
- reduces missed short vocalizations.

---

## Mel Spectrogram Configuration

```python
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 16000
```

---

# Spectrogram Pipeline

```python
waveform
   ↓
resample
   ↓
normalize
   ↓
log-mel spectrogram
   ↓
SpecAugment
   ↓
model input
```

---

# Recommended Models

## Best Starting Models

### EfficientNet
Recommended:
- EfficientNet-B0
- EfficientNet-B2

Advantages:
- strong baseline,
- efficient training,
- proven BirdCLEF success.

---

### ConvNeXt
Recommended:
- ConvNeXt-Tiny

Advantages:
- stronger feature extraction,
- better optimization stability,
- modern convolutional architecture.

---

# Output Head

Final layer:

```python
Linear(hidden_dim, num_classes)
```

Activation:

```python
sigmoid()
```

Reason:
BirdCLEF is multi-label.

Softmax is incorrect because:
- multiple species can exist simultaneously.

---

# Loss Function

## Recommended

```python
BCEWithLogitsLoss(pos_weight=...)
```

---

## Better Alternative

### Focal BCE

Useful because:
- dataset is heavily imbalanced,
- rare species require stronger gradients.

---

# Data Augmentation

## Essential Augmentations

### 1. SpecAugment
- frequency masking,
- time masking.

Improves:
- robustness,
- generalization.

---

### 2. Mixup
Mix spectrograms from multiple species.

Improves:
- regularization,
- minority-class robustness.

---

### 3. Background Mixing
Mix:
- rain,
- insects,
- wind,
- environmental soundscapes.

Improves:
- real-world soundscape robustness.

---

### 4. Gain Scaling
Random amplitude adjustment.

Simulates:
- recorder variability,
- distance variation.

---

# Cross Validation Strategy

## Use Multilabel Stratification

Recommended:
```python
MultilabelStratifiedKFold
```

Reason:
- preserves label distribution,
- avoids fold imbalance.

---

# Important Leakage Prevention

Group by:
```python
soundscape_filename
```

This prevents:
- temporal leakage,
- overlapping-window leakage,
- artificially inflated validation scores.

---

# Inference Pipeline

## Sliding Window Inference

Example:
```text
0–5s
2.5–7.5s
5–10s
7.5–12.5s
...
```

Each window generates:
- independent probabilities.

---

# Aggregation Methods

## Recommended
### Max Pooling
```python
final_pred = window_preds.max(axis=0)
```

Captures:
- short rare calls,
- transient events.

---

## Alternative
### Mean Pooling
More stable but may suppress rare weak events.

---

# Threshold Calibration

## Very Important

Do NOT use:
```python
threshold = 0.5
```

Instead:
- optimize per-class thresholds,
- maximize validation LWLRAP/F1.

---

# Recommended Metrics

## Primary
- LWLRAP
- mAP
- Macro F1

---

## Secondary
- PR-AUC
- per-class AP

## Competition metrics
- The evaluation metric for this contest is a version of macro-averaged ROC-AUC that skips classes that have no true positive labels.

---

# Training Recipe

## Suggested Hyperparameters

```python
batch_size = 32
epochs = 20–40
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
```

Scheduler:
```python
CosineAnnealingLR
```

---

# Recommended Training Tricks

## Mixed Precision
```python
torch.cuda.amp
```

Reduces:
- VRAM usage,
- training time.

---

## EMA (Exponential Moving Average)

Improves:
- inference stability,
- leaderboard consistency.

---

# Test-Time Augmentation (TTA)

## Recommended TTA
- overlapping windows,
- gain shifts,
- spectrogram shifts.

Final prediction:
```python
mean(predictions)
```

---

# Expected Strengths

This pipeline:
- trains relatively fast,
- is highly scalable,
- handles multi-label audio well,
- provides strong leaderboard baselines,
- supports stacking later.

---

# Expected Weaknesses

Limitations:
- weaker long-range temporal modeling,
- CNN-only representation,
- limited contextual reasoning compared to transformers.

---

# Upgrade Path

After stabilizing this pipeline:

## Step 1
Add:
- stronger augmentations,
- threshold optimization,
- TTA.

---

## Step 2
Upgrade backbone:
- HTS-AT,
- AST,
- PaSST.

---

## Step 3
Add:
- pseudo-labeling,
- SSL pretraining,
- stacking ensembles.

---

# Recommended Tech Stack

## Core Libraries
- PyTorch
- timm
- torchaudio
- librosa
- albumentations

---

## Experiment Tracking
- Weights & Biases
- MLflow

---

# Final Notes

This is the highest-value first pipeline because it directly addresses:
- multi-label soundscapes,
- class imbalance,
- noisy field recordings,
- overlapping species,
- real BirdCLEF inference behavior.

It also forms the foundation for:
- transformer hybrids,
- stacking,
- pseudo-labeling,
- self-supervised learning,
- and ensemble systems later.