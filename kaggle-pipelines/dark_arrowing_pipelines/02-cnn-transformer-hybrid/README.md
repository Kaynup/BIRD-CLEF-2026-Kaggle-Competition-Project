CNN + Transformer Hybrid (BEST HIGH-END MODEL)

## Why This Pipeline Matters

EDA revealed:
- temporal vocal structures,
- repetitive chirps,
- long-duration context dependencies.

CNNs capture:
- local spectral patterns.

Transformers capture:
- temporal relationships,
- long-range context,
- overlapping acoustic events.

---

## Pipeline Structure

```text
Mel Spectrogram
      ↓
CNN Feature Extractor
      ↓
Spectrogram Tokens
      ↓
Transformer Encoder
      ↓
Multi-label Sigmoid Head
```

---

## Recommended Models

### CNN Frontend
- EfficientNet
- ConvNeXt
- ResNet

### Transformer
- AST
- HTS-AT
- PaSST

---

## Why Winners Use This

BirdCLEF winners heavily use:
- AST,
- HTS-AT,
- PaSST,
- Transformer-based audio encoders. :contentReference[oaicite:1]{index=1}

Because:
- bird calls have strong temporal structure,
- overlapping sound events require attention mechanisms.

---

## Key Advantages
- handles overlapping species better,
- learns long temporal dependencies,
- strong generalization.

---

## Main Weakness
- expensive training,
- high VRAM usage,
- slower experimentation.

---

## Recommended Usage
Use this after:
- strong baseline stabilization,
- augmentation tuning,
- threshold calibration.

---