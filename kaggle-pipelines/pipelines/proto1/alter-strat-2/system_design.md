# System Design: Alter-Strat-2 (Geo-Acoustic Self-Distillation)

## 1. Core Philosophy
Your intuition is exactly correct: the previous pipeline showed that the **mapping between coordinate space and species density is extremely powerful** for solving BirdCLEF. Since the Kaggle test set doesn't provide these coordinates, we must bridge the gap mathematically. If our **Head B (Geolocation Regressor)** can accurately predict the location just by listening to the ecosystem (wind, insects, background noise), we can feed that predicted location into our pre-trained KNN. This will recreate the powerful validation performance you saw earlier, but in a completely legitimate, zero-leakage way that will translate directly to the Leaderboard.

## 2. Acoustic Fundamentals (Inherited from 01-multilabel-effnet-convnext)
Before applying any spatial tricks, this pipeline adopts the flawless acoustic processing fundamentals that scored ~0.82+:
*   **Sliding Window Inference**: We use 5-second windows with a **2.5-second overlap** during inference to ensure no calls are cut off.
*   **Max-Pooling Aggregation**: We use `max(axis=0)` across window predictions to ensure short, rare calls aren't diluted.
*   **True Multi-Label Head**: We use a `Sigmoid` output with `BCEWithLogitsLoss` to handle overlapping species.
*   **Heavy Background Augmentation**: We mix rain, wind, and insect noise into training to make the model robust to real-world soundscapes.

## 3. Model Architecture (The Dual-Head ViT)
The core model is a Vision Transformer (ViT) with a customized dual-head top:
*   **Backbone**: `vit_base_patch16_224` (or similar) taking in 5-second Mel-spectrograms.
*   **Head A (Acoustic Classifier)**:
    *   Architecture: `Linear(in_features, 512) -> ReLU -> Dropout -> Linear(512, NUM_CLASSES)`
    *   Output: `[Batch, 234]` - The acoustic probability of each bird species.
*   **Head B (Geolocation Regressor)**:
    *   Architecture: `Linear(in_features, 512) -> ReLU -> Dropout -> Linear(512, 2)`
    *   Output: `[Batch, 2]` - The predicted `[Latitude, Longitude]` in radians.

## 4. Loss Function & Training Dynamics
The network is optimized jointly using a multi-task loss function:

`Total_Loss = BCEWithLogitsLoss(Head_A_Preds, True_Species) + λ * MSELoss(Head_B_Preds, True_Coordinates)`

*   **λ (Lambda)**: A hyperparameter (e.g., `10.0` or `1.0` depending on scaling) to ensure the MSE of the coordinates is balanced with the BCE of the species.
*   **Data Augmentation**: We will use random 5-second crops from the training files to ensure the model doesn't just learn the first 5 seconds of empty noise.

## 5. Confidence-Based Blending (Inference)
As you brilliantly suggested, we shouldn't blindly blend the KNN prior if the acoustic model is already highly confident. We introduce **Confidence-Gated Blending**.

*   `CONFIDENCE_THRESHOLD`: A global toggle/variable (e.g., `0.85`).
*   If `ViT_Acoustic_Prob > CONFIDENCE_THRESHOLD`: We trust the ViT completely. `Final_Prob = ViT_Acoustic_Prob`.
*   If `ViT_Acoustic_Prob <= CONFIDENCE_THRESHOLD`: The model is unsure, so we fall back on the ecosystem prior. `Final_Prob = (ViT_Acoustic_Prob * (1 - α)) + (KNN_Prior * α)`.

## 6. Cross-Validation Strategy
To guarantee zero-leakage, we must group our k-folds by **Author** (Recordist). Grouping by `filename` is too weak because a single author might record 10 files in the exact same location in 10 minutes. By grouping by author, we force the validation set to test entirely unseen environments and locations.
