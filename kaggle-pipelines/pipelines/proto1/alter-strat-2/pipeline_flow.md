# Pipeline Flow: Alter-Strat-2

This document illustrates the flow of the Geo-Acoustic Self-Distillation pipeline, specifically highlighting the Dual-Head ViT architecture and the Confidence-Based Blending during inference.

## Training Pipeline Flow

```mermaid
graph TD
    A[Raw Audio] --> B(Mel-Spectrogram)
    B --> C{Vision Transformer Backbone}
    
    C --> D[Head A: Acoustic Classifier]
    C --> E[Head B: Geolocation Regressor]
    
    D -->|BCE Loss| F((Total Loss))
    E -->|MSE Loss| F
    
    G[True Species Labels] --> D
    H[True Coordinates] --> E
    H --> I[Train KNN Spatial Regressor]
    G --> I
```

## Inference Pipeline Flow (Zero-Leakage Test Set)

```mermaid
graph TD
    A[Unseen Test Soundscape] --> B(Mel-Spectrogram)
    B --> C{Trained ViT Backbone}
    
    C --> D[Head A: Acoustic Probs]
    C --> E[Head B: Predicted Coordinates]
    
    E -->|Feed Predicted Lat/Lon| F{Pre-trained Spatial KNN}
    F -->|Output| G[Spatial Prior Probs]
    
    D --> H{Confidence Gating}
    
    H -->|Max Prob > Threshold| I[Use Pure ViT Acoustic Probs]
    H -->|Max Prob <= Threshold| J[Blend ViT Probs & KNN Prior]
    G --> J
    
    I --> K((Final Submission Probs))
    J --> K
```

## Confidence-Gated Blending Logic (Pseudocode)

```python
# Inference-time blending decision
THRESHOLD = 0.85
ALPHA = 0.3  # Weight of the KNN prior

final_probs = []
for i in range(batch_size):
    vit_prob = vit_acoustic_probs[i]
    if torch.max(vit_prob) > THRESHOLD:
        # The model is highly confident it heard the bird
        final_probs.append(vit_prob)
    else:
        # The model is unsure, so we lean on the environmental prior
        knn_prob = knn_spatial_priors[i]
        blended = (vit_prob * (1.0 - ALPHA)) + (knn_prob * ALPHA)
        final_probs.append(blended)
```
