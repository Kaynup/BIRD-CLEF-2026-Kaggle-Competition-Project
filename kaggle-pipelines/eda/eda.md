
# Exploratory Data Analysis (EDA) Interpretation Notes

## 1. Dataset Overview

### Training Metadata (`train.csv`)
- Shape: **35,549 rows × 15 columns**
- Unique `primary_label` values: **206 species/classes**
- Key columns:
  - `primary_label`
  - `secondary_labels`
  - `latitude`, `longitude`
  - `scientific_name`
  - `common_name`
  - `class_name`
  - `rating`
  - `filename`

### Interpretation
This is a moderately large multi-class bioacoustic dataset with:
- strong ecological diversity,
- geographic variation,
- variable recording quality,
- and heavy class imbalance.

The metadata indicates that the dataset combines:
- bird calls,
- frogs,
- insects,
- mammals,
- reptiles.

This means the classification problem is not only **multi-class**, but also partially **cross-taxonomic acoustic recognition**.

---

# 2. Geographic Distribution

## Latitude / Longitude Statistics

### Observed Distribution
- Mean latitude: approximately `-8.16`
- Mean longitude: approximately `-60.74`
- Wide standard deviation in both coordinates.

### Interpretation
The recordings are geographically spread across multiple ecological zones.

This matters because:
- species vocalizations vary regionally,
- background noise differs by biome,
- habitat acoustics affect frequency propagation.

### Practical Use
Geolocation can be used as:
- a metadata feature,
- a prior probability estimator,
- or for geographic filtering during inference.

Example:
- If a species only occurs in a specific biome,
  predictions outside that biome can be down-weighted.

Potential future feature engineering:
- clustering by eco-region,
- latitude/longitude embeddings,
- seasonal-geographic priors.

---

# 3. Class Distribution (Primary Labels)

## Species Frequency Distribution

### Top species counts
Examples:
- `rubthr1`: 499 samples
- `banana`: 498
- `fepowl`: 497
- lowest classes: only 1 sample

### Statistics
- Mean files/species: 172.6
- Median: 125
- Standard deviation: 155
- Min: 1
- Max: 499
- Imbalance ratio: 499× between largest and smallest classes

---

## Interpretation of the Distribution

This is a **heavily long-tailed distribution**.

### What it means
A small number of species dominate the dataset while many species are rare.

This creates:
- representation bias,
- unstable gradients,
- poor minority-class recall,
- overfitting to dominant species.

---

## Practical Implications

### Without correction:
The model will:
- predict common species excessively,
- ignore rare taxa,
- achieve misleadingly high accuracy.

---

## Recommended Techniques

### Loss Functions
Use:
- Focal Loss
- Class-balanced loss
- Weighted BCE / CE

### Sampling
Use:
- WeightedRandomSampler
- oversampling of minority classes
- stratified batching

### Augmentations
Rare classes especially benefit from:
- pitch shifting,
- time stretching,
- background mixing,
- SpecAugment.

### Evaluation
Avoid relying only on accuracy.

Prefer:
- macro F1,
- mAP,
- LWLRAP,
- per-class recall.

---

# 4. Soundscape Labels Analysis

## Soundscape Dataset
- 66 unique soundscape recordings
- 1,478 annotated 5-second windows
- Total taxa occurrences: 6,244

### Average Species Per Window
- Mean: 4.22 species/window
- Maximum: 10 species/window
- Multi-label windows: 89.4%

---

# Interpretation

This confirms the problem is fundamentally:
## Multi-label acoustic event detection

Not simple single-label classification.

A single soundscape segment may contain:
- multiple birds,
- insects,
- frogs,
- overlapping harmonics,
- environmental noise.

---

# Implications for Modeling

## The model architecture should support:
- sigmoid outputs,
- multi-label objectives,
- threshold calibration.

Do NOT use:
- softmax-only classification.

---

# Recommended Metrics
Use:
- mAP
- LWLRAP
- PR-AUC
- macro F1

These are more appropriate than standard accuracy.

---

# 5. Taxonomic Distribution

## Taxonomy Statistics
- Total taxonomy species: 234
- Species with training audio: 206
- Coverage: 88%

### Class Distribution
- Aves: 162
- Amphibia: 35
- Insecta: 28
- Mammalia: 8
- Reptilia: 1

---

# Interpretation

The dataset is bird-dominated.

This introduces:
- taxonomic imbalance,
- acoustic-domain imbalance.

Bird calls are generally:
- harmonic,
- tonal,
- frequency structured.

Insects and amphibians often exhibit:
- broadband textures,
- repetitive chirps,
- noisy spectral patterns.

---

# Practical Use

You can:
- build taxon-aware models,
- use hierarchical classification,
- add class-type embeddings.

Example:
1. First classify:
   - bird,
   - frog,
   - insect,
   - mammal.

2. Then classify species within the taxon.

This hierarchical approach can improve robustness.

---

# 6. Audio File Distribution

## Audio Statistics
- Total training audio files: 35,549
- Format: `.ogg`
- Total size: 10.75 GB
- Average file size: 0.30 MB

### Soundscape Files
- Total size: 5.38 GB
- Average size: 0.50 MB

---

# Interpretation

The dataset is relatively efficient in storage,
but large enough for deep learning training.

---

# Engineering Implications

## Storage / Loading
Efficient pipelines are necessary:
- lazy loading,
- streaming,
- caching,
- parallel decoding.

---

## Recommended Libraries
- `torchaudio`
- `librosa`
- `soundfile`
- `ffmpeg`

---

# 7. Audio Duration & Sampling Analysis

Random audio files were analyzed.

The notebook inspected:
- sample rates,
- durations,
- spectrogram properties.

---

# Interpretation

Audio heterogeneity is expected:
- varying duration,
- varying noise floors,
- recording device differences,
- environmental variance.

This variability improves generalization potential,
but complicates preprocessing.

---

# Recommended Standardization

## Typical Pipeline
- resample to fixed rate (e.g. 32kHz)
- normalize amplitude
- fixed-duration cropping
- mel spectrogram conversion

---

# 8. Spectrogram Visualizations

The notebook generated mel spectrograms for selected species.

---

# Interpretation of Spectrograms

Spectrograms reveal:
- harmonic bands,
- temporal pulse structure,
- frequency occupancy,
- rhythmic signatures.

These are the main features learned by CNNs.

---

# What We Can Learn from the Spectrograms

## Birds
Usually show:
- harmonic stacks,
- frequency sweeps,
- structured melodic contours.

## Frogs
Often exhibit:
- repetitive low-frequency pulses,
- dense temporal blocks.

## Insects
Typically:
- high-frequency narrow-band energy,
- repetitive chirping patterns.

---

# Practical Modeling Insights

Different taxa occupy different spectral regions.

This suggests:
- frequency-aware augmentations,
- adaptive bandpass filtering,
- multi-resolution spectrograms,
- taxon-specific preprocessing.

---

# 9. Distribution Visualizations

The notebook plotted:
- class distributions,
- label frequency histograms,
- multi-label segment distributions.

---

# How to Interpret These Distributions

## Long-Tail Distribution
Indicates:
- imbalance,
- sparse minority representation,
- difficult generalization.

### Usefulness
Helps determine:
- augmentation strategy,
- resampling strategy,
- weighting strategy.

---

## Multi-Label Count Distribution
Most windows contain 4–6 species.

### Meaning
The environment is acoustically dense.

The model must learn:
- overlapping signals,
- partial masking,
- co-occurrence patterns.

---

# Practical Improvements from This Insight

## Threshold Tuning
Different species require different thresholds.

Instead of:
```python
threshold = 0.5
```

Use:
- per-class thresholds,
- validation-optimized calibration.

---

## Co-occurrence Modeling
Species appearing together frequently can improve prediction.

Possible methods:
- graph neural networks,
- label correlation matrices,
- conditional decoding.

---

# 10. Key Dataset Challenges

## 1. Severe Class Imbalance
Requires:
- weighted training,
- focal loss,
- augmentation.

---

## 2. Multi-Label Complexity
Requires:
- sigmoid outputs,
- robust thresholding,
- ranking-based metrics.

---

## 3. Acoustic Noise
Field recordings contain:
- wind,
- rain,
- insects,
- overlapping calls,
- anthropogenic noise.

Noise robustness becomes critical.

---

## 4. Domain Shift
Train clips and soundscapes differ significantly.

Need:
- strong augmentation,
- domain adaptation,
- robust validation.

---

# 11. Recommended Modeling Pipeline

## Preprocessing
- resample audio,
- mel spectrogram generation,
- normalization,
- SpecAugment.

---

## Model Candidates
### CNNs
- EfficientNet
- ConvNeXt
- ResNet

### Audio Transformers
- AST
- BEATs
- HTS-AT

### Hybrid Models
CNN + Transformer.

---

## Training Strategy
- stratified folds,
- class-balanced sampling,
- mixed precision training,
- early stopping,
- threshold optimization.

---

# 12. Most Important EDA Conclusions

## The dataset is:
- highly imbalanced,
- strongly multi-label,
- acoustically heterogeneous,
- geographically diverse,
- taxonomically diverse.

---

## Therefore the solution should:
- use multi-label learning,
- handle long-tail imbalance,
- exploit spectrogram structure,
- leverage metadata,
- optimize ranking metrics instead of accuracy.

---

# 13. Final Takeaways

The EDA strongly suggests that success will depend more on:
- handling imbalance,
- robust augmentation,
- threshold calibration,
- and representation learning

than on simply increasing model size.

The spectrogram distributions and label statistics provide direct guidance for:
- architecture selection,
- loss engineering,
- augmentation policies,
- and evaluation methodology.
