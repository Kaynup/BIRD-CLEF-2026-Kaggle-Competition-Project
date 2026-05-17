Two-Stage SED + Species Classifier (MOST SCIENTIFICALLY ROBUST)

## Why This Pipeline Matters

EDA showed:
- soundscapes contain massive background noise,
- inactive regions dominate long recordings,
- overlapping environmental sounds exist.

Instead of classifying the entire recording:
first detect acoustic events.

---

## Pipeline Structure

```text
Long Audio
     ↓
Sound Event Detector (SED)
     ↓
Extract Active Regions
     ↓
Species Classifier
     ↓
Aggregate Predictions
```

---

## Stage 1 — Event Detection

Detect:
- chirps,
- calls,
- pulses,
- acoustic activity regions.

Models:
- U-Net
- CRNN
- lightweight CNN detectors

---

## Stage 2 — Species Classification

Classify extracted segments using:
- EfficientNet
- ConvNeXt
- HTS-AT

---

## Why This Is Powerful

Removes:
- silence,
- wind,
- rain,
- irrelevant background noise.

Improves:
- signal-to-noise ratio,
- rare-event focus,
- classifier precision.

---

## Why It Matches the EDA

EDA strongly indicated:
- noisy field recordings,
- overlapping calls,
- environmental interference.

This pipeline explicitly models those issues.

---
