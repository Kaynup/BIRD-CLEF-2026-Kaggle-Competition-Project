# Pipeline ideas for BirdCLEF 2026 — CNN / audio pipelines

Summary: this document lists many experiment pipelines focused on strong generalization, stratified CV, OOF, reproducibility, and aggressive experimentation. Use this as a cookbook: pick a pipeline, implement config-driven notebooks, track runs, produce OOF and final submission CSVs.

**Dataset analysis (quick observations)**
- Multi-source recordings: `iNat`, `XC` (xeno-canto), and curated datasets appear in `data/raw/train.csv` and `data/raw/taxonomy.csv`.
- Many taxonomic classes appear (Amphibia, Aves, Insecta, Mammalia) — not only birds; taxonomy mapping contains sonotypes and grouped entries.
- Training has two modalities:
  - short single-label recordings (file-level labels in `train.csv`) — varied durations and sources
  - long `soundscapes` (sliding 5s segments in `train_soundscapes_labels.csv`) with multi-label species per 5s window
- Geographic focus: Pantanal / broader South America coordinates; recorder sites listed in `recording_location.txt`.
- Label imbalance and label noise present (`uncertain` flags, `advertisement call`, etc.).

Principles that should apply everywhere:
- Use config files (TOML/YAML) with a single `train_config` object for reproducibility.
- Seed all libraries and record seeds in the run metadata.
- Produce OOF predictions and per-fold checkpoints; save fold-level metrics and confusion matrices.
- Use stratified CV designed for multi-label: `StratifiedGroupKFold` or `iterative-stratification` (multilabel stratifier) with groups = soundscape or recorder-location to prevent leakage.
- Use strong data augmentation (SpecAugment, time stretch, pitch shift, background mix) and test-time augmentation (TTA).

Pipelines (prioritized, start small then scale):

1) Baseline: Mel-spectrogram + ResNet50 (single-label training)
- Input: 3s/5s log-mel (128 bins), augmentation: random crop, time masking
- Model: pretrained ResNet50 on ImageNet (replace first conv for single-channel or stack mel channels)
- Loss: CrossEntropy (class-weighting or focal if imbalance)
- CV: Stratified K-Fold on `primary_label` (for single-label items only)
- OOF: per-file predictions; simple softmax thresholding.

2) Soundscape multi-label classifier (sliding-window) — strong baseline
- Input: 5s log-mel (same config as dataset segments)
- Model: EfficientNet / ConvNeXt or SEResNet backbone adapted for multi-label (sigmoid final)
- Loss: BCEWithLogits + positive-class weighting; add label smoothing
- CV: iterative stratification on multi-label vectors with `group` = soundscape file (prevent time leakage)
- Inference: sliding windows at inference; aggregate via max and noise-robust pooling

3) CNN + Transformer (spectrogram tokens)
- Frontend: small CNN encoder to produce token sequence
- Encoder: lightweight Transformer (4-8 layers) to capture temporal context
- Benefit: longer context modeling per 5–30s clip, better at overlapping calls

4) Pretrain on in-domain unlabeled spectrograms (self-supervised)
- Method: SimCLR / BYOL / DINO on mel patches from all training audio + soundscapes
- Use learned encoder as backbone for downstream classifier — fine-tune with small LR

5) Multi-task: species classification + call-type / confidence regression
- Add extra heads: (a) call-type classifier (advertisement/chorus/etc when present), (b) confidence/regression for `rating`
- Shared encoder helps regularize and leverage auxiliary labels in `train.csv`

6) Pseudo-labeling & semi-supervised (large scale)
- Train teacher on labeled set + soundscapes; infer on unlabeled segments or test set; select high-confidence predictions (thresholded) and retrain student with mix of labeled + pseudo-labeled data

7) Two-stage pipeline: SED (sound event detection) + species classifier
- Stage 1: SED model (U-Net / CNN) to detect bird-syllable segments in long audio
- Stage 2: classify detected event crops with a strong classifier (EfficientNet/ECA-ResNet)
- Useful to remove background noise and focus on events for rare species

8) Embedding + nearest-neighbor / metric-learning ensemble
- Train embedding model with ArcFace/Triplet/Contrastive losses on mel patches
- At inference, use kNN on embeddings and mix kNN scores with classifier probabilities

9) Mixup / CutMix / SpecAugment heavy-augmentation pipeline
- Aggressive augmentation schedule during training
- Use Mixup + Balanced sampling to reduce overfitting to common species

10) Curriculum / progressive augmentation
- Start training on clean single-label recordings, then progressively add noisy/external (xeno-canto) and soundscape segments
- Can speed convergence and improve robustness

11) Time-aware models and seasonal/location features
- Append metadata features (latitude, longitude, month, recorder id) to classifier through FiLM layers or tabular heads
- Use these as regularizers or priors (but validate carefully to avoid leakage)

12) Multi-resolution ensemble (short + long context)
- Train models on different input durations (1s, 5s, 30s). Ensemble by stacking or average pooling of predictions across durations for final submission.

13) Learnable frontend (LEAF) / waveform models
- Try LEAF or wav2vec-like frontend + small CNN to see whether waveform-level features beat log-mel for specific taxa

14) Meta-learning / few-shot for rare species
- Prototype embeddings + prototypical networks trained on classes with few examples; useful when some species have <10 samples

15) Transformer-only audio classifier (vision-transformer on spectrogram patches)
- Strong potential with large compute; requires SSL pretraining on spectrograms to be competitive

16) Autotune hyperparameter framework
- Integrate `Optuna` or `Ray Tune` to optimize: LR, weight decay, mixup alpha, mel bin count, model depth

17) Efficient inference and distillation
- Distill large ensemble into a single smaller student model for faster inference and submission generation
- Use quantization + pruning for submission-time speed

18) Calibration & threshold tuning pipeline
- Calibrate per-class thresholds on OOF validation to maximize the metric (e.g., mAP/F1 or Kaggle metric)

19) Stacking & OOF stacking pipeline
- Save OOF preds from many models, train a small meta-learner (lightGBM / logistic) on OOF features, and apply to test set

20) Robustness-focused experiments
- Background mixing with real soundscapes, SNR sweeps, and adversarial noise augmentation

Implementation notes and recommended tooling
- Frameworks: PyTorch Lightning or Lightning Flash for experiment structure; Hugging Face Transformers for pretrained backbones where appropriate.
- Data pipeline: torchaudio + librosa for mel extraction; cache mel features on-disk (LMDB or Parquet) for fast experiments.
- CV details:
  - For single-label files: StratifiedKFold by `primary_label`.
  - For multi-label soundscapes: use `iterative_train_test_split` (multilabel) or `MultilabelStratifiedKFold` from `iterative-stratification`, grouping by `filename` to avoid leakage.
  - For time/location grouping: create group keys like `site_date` or `recorder_id` when available.
- OOF: always save fold-level OOF CSVs and per-fold predictions (probabilities) to reproduce stacking.
- Metrics: report per-fold macro-F1, micro-F1, AUPRC; also compute per-class AP to monitor rare-class performance.
- Reproducibility:
  - `seed_everything(seed, workers=True)` in Lightning; record `torch.get_rng_state()` and `numpy` seed.
  - Save full run config, git commit hash, and conda environment (`pip freeze > requirements.txt`).
- Experiment tracking: use Weights & Biases or MLflow; record artifacts: model checkpoints, OOF preds, confusion matrices, and sample visualizations.

Quick experiment ladder (how to iterate fast):
- Step 0: small prototype notebook: 1x GPU, train ResNet50 on mel patches for 5 epochs. Validate CV split = 1 fold. (fast feedback)
- Step 1: enable iterative stratified K=5 soundscape CV, add SpecAugment and test mixup.
- Step 2: scale to EfficientNet + fine-tune from pretrained SSL encoder.
- Step 3: run pseudo-labeling and stacking; produce OOF and final submission.

Next steps I can do for you (choose):
- Implement a starter notebook for Pipeline 1 (baseline) with config, data loader, CV split, and training loop.
- Scaffold the multi-label soundscape notebook (Pipeline 2) with CV and OOF saving.
- Create a set of small scripts to extract and cache mel-spectrograms for fast iteration.

---
Small notes: I inspected `data/raw/train.csv`, `data/raw/taxonomy.csv`, and `data/raw/train_soundscapes_labels.csv` to inform the pipelines. If you want, I can now scaffold one or more of these pipelines into runnable notebooks under `kaggle-notebooks/cnn/`.
