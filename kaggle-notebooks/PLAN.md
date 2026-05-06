The error is caused by this line:

probs = np.stack([r[1] for r in all_rows], axis=0)

all_rows is empty.

The real root issue appears earlier:

Test soundscapes found: 0

So:

test_files = sorted(TEST_SND.glob('*.ogg'))

found no files.

That means your inference pipeline never generated predictions.

Why This Happens

In BirdCLEF competitions, Kaggle behaves differently depending on environment:

During notebook development:
test_soundscapes/ may be empty.
During submission:
Kaggle injects hidden test files.

So locally in notebook execution:

len(test_files) == 0

is expected.

Immediate Fix

Replace this section:

test_files = sorted(TEST_SND.glob('*.ogg'))
print(f'Test soundscapes found: {len(test_files)}')

with:

test_files = sorted(TEST_SND.glob('*.ogg'))

if len(test_files) == 0:
    print("No test files found. Using sample submission fallback.")
    
    sub = sample_sub.copy()

    baseline_prob = 1.0 / n_classes

    for sp in SPECIES:
        sub[sp] = baseline_prob

    submission_path = OUT / 'submission.csv'
    sub.to_csv(submission_path, index=False)

    print(f'Fallback submission saved: {submission_path}')
else:
    print(f'Test soundscapes found: {len(test_files)}')

This prevents the crash.

Secondary Fix

You also need protection here:

if len(all_rows) == 0:
    print("No predictions generated.")
    probs = np.empty((0, n_classes))
else:
    probs = np.stack([r[1] for r in all_rows], axis=0)

Without this, any inference failure crashes the notebook.

Bigger Problem in Your Pipeline

Your current pipeline is fundamentally:

log-mel statistics
→ RandomForest
→ single-label approximation

This is a baseline only.

Your OOF ROC-AUC:

0.8729

looks deceptively good because:

strong leakage risk
clip-level overlap
weak validation realism
single-label expansion inflates metrics

On leaderboard this architecture will collapse against CNNs.

Major Technical Weaknesses
1. Random Forest Is Wrong Tool

RF does not understand:

temporal dynamics
harmonics
bird call structure
spectrogram geometry

BirdCLEF winners use:

EfficientNet
ConvNeXt
AST
PaSST
HTS-AT
BirdNET-style encoders
2. Feature Compression Is Too Aggressive

You reduce spectrogram to:

mean + std

destroying:

chirp timing
modulation
rhythm
harmonic contours

You compressed:

(64 mel × ~300 frames)
→ 128 numbers

This discards nearly all acoustic structure.

3. Fake Multi-Label Handling

This part:

expanded them above → each row is one species

is mathematically wrong for soundscapes.

BirdCLEF is multilabel.

A segment may contain:

insects
frogs
birds
wind
overlapping calls

You converted multilabel into duplicated single-label samples.

That biases probability estimation badly.

How To Go "Full and All Out"

You need a competition-grade pipeline.

Recommended Modern Architecture
Stage 1 — Better Audio Representation

Move from:

summary statistics

to:

full spectrogram tensors

Use:

torchaudio.transforms.MelSpectrogram

or

nnAudio

Target:

128 mel bins
5 sec windows
32000 Hz

Output:

[1, 128, time]
Stage 2 — Deep Learning Backbone

Use:

Best practical option
EfficientNet-B0/B2

via:

timm.create_model()

Why:

strong BirdCLEF history
fast
stable
good Kaggle GPU fit
Better Options
HTS-AT

Transformer for audio tagging.

Very strong for multilabel soundscapes.

PaSST

State-of-the-art audio spectrogram transformer.

Stage 3 — True Multi-Label Learning

Replace:

RandomForestClassifier

with:

BCEWithLogitsLoss

Target tensor:

[batch, 234]

not single integer labels.

Stage 4 — Proper Validation

Current validation leaks heavily.

You need:

Grouped Stratification

Group by:

recording_id
filename
location

Use:

StratifiedGroupKFold

Otherwise same acoustic environment leaks across folds.

Stage 5 — Augmentation

Critical for BirdCLEF.

Use:

Spectrogram augmentations
SpecAugment
time masking
frequency masking
Audio augmentations
background noise mixing
random gain
pitch shift
time stretch
mixup
cutmix

Libraries:

audiomentations
Stage 6 — Better Inference

Current:

5-second independent windows

Weak.

Use:

Sliding Window Inference

Example:

window = 5s
hop = 1s

Then aggregate:

max pooling
geometric mean
attention pooling
Stage 7 — Threshold Optimization

Huge leaderboard gains come here.

Instead of:

prob > 0.5

optimize per-class thresholds:

best_thresh[class_id]

using OOF predictions.

Stage 8 — Pseudo Labeling

Top BirdCLEF teams do:

train
→ infer on unlabeled soundscapes
→ keep confident predictions
→ retrain

Massive gains.

Stage 9 — Ensemble

Final systems usually combine:

EfficientNet
ConvNeXt
Transformer
different folds
different mel configs

Then average logits.

Practical "Full-Out" Kaggle Stack

A strong realistic stack:

Audio:
- 32kHz
- 128 mel
- 5 sec windows

Model:
- EfficientNet-B2

Training:
- BCEWithLogitsLoss
- AMP mixed precision
- AdamW
- Cosine scheduler

Aug:
- SpecAugment
- Mixup
- Background noise

Validation:
- StratifiedGroupKFold

Inference:
- sliding windows
- TTA

Postprocessing:
- threshold tuning
- class priors
Expected Performance Jump

Your current RF baseline:

LB ~0.45–0.60 likely

A proper CNN pipeline:

0.75–0.90+

depending on engineering quality.

Most Important Upgrade Order

Do these first:

Replace RF with CNN
Use full spectrograms
True multilabel targets
Proper grouped CV
Augmentations
Threshold tuning

These alone can transform the system.

Another Critical Bug

This line is dangerous:

duration = len(y_full) / SR
n_segments = int(duration // SEGMENT_SEC)

If duration slightly underflows:

59.999 sec

you lose final segment.

Safer:

n_segments = max(1, int(np.ceil(duration / SEGMENT_SEC)))
Final Note

Your pipeline is actually well-structured for a baseline:

clean organization
efficient caching
sensible feature extraction
proper OOF logic