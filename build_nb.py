import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [line + "\n" for line in text.split("\n")]})

add_md("# BirdCLEF 2026 - PyTorch CNN Pipeline (No Leakage, Multi-label, CNN)\nBased on PLAN.md recommendations.")

add_code("""import os, gc, random, warnings, time
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import timm

warnings.filterwarnings('ignore')
print('Libraries loaded')
""")

add_code("""# ── Config ─────────────────────────────────────────────────────────────
BASE = Path('/kaggle/input/competitions/birdclef-2026')
TRAIN_AUDIO = BASE / 'train_audio'
TRAIN_SND = BASE / 'train_soundscapes'
TEST_SND = BASE / 'test_soundscapes'
OUT = Path('/kaggle/working')

SR = 32_000
SEGMENT_SEC = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 50
FMAX = 14_000

MAX_CLIPS_PER_SPECIES = 30
MIN_RATING = 3.0
N_FOLDS = 5
SEED = 42

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
""")

add_md("## 1. Load Data & Taxonomy")

add_code("""train_df = pd.read_csv(BASE / 'train.csv')
taxonomy = pd.read_csv(BASE / 'taxonomy.csv')
snd_labels = pd.read_csv(BASE / 'train_soundscapes_labels.csv')
sample_sub = pd.read_csv(BASE / 'sample_submission.csv')

SPECIES = [c for c in sample_sub.columns if c != 'row_id']
n_classes = len(SPECIES)
species_to_idx = {sp: i for i, sp in enumerate(SPECIES)}
idx_to_species = {i: sp for i, sp in enumerate(SPECIES)}

print(f'Classes: {n_classes}')
""")

add_md("## 2. Prepare Data (No Leakage, True Multi-Label, Grouping)")

add_code("""# Filter train_audio
target_set = set(SPECIES)
train_df = train_df[train_df['primary_label'].isin(target_set)].copy()
train_df = train_df[(train_df['rating'] == 0) | (train_df['rating'] >= MIN_RATING)].copy()

if MAX_CLIPS_PER_SPECIES:
    train_df = (
        train_df.groupby('primary_label', group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_CLIPS_PER_SPECIES), random_state=SEED))
    ).reset_index(drop=True)

# Create targets for train_audio (single label becomes one-hot in multi-label context)
train_df['filepath'] = train_df['filename'].apply(lambda x: str(TRAIN_AUDIO / x))
train_df['group'] = train_df['filename'] # Grouping by filename
train_df['start_sec'] = 0.0 # Will dynamically slice in dataset
train_df['is_soundscape'] = False

# Soundscapes
def seconds_from_hms(hms_str):
    h, m, s = map(int, str(hms_str).split(':'))
    return h*3600 + m*60 + s

snd_labels['start_sec'] = snd_labels['start'].apply(seconds_from_hms)
snd_labels['filepath'] = snd_labels['filename'].apply(lambda x: str(TRAIN_SND / x))
snd_labels['group'] = snd_labels['filename']
snd_labels['is_soundscape'] = True

# Convert semicolon-separated primary_label to multi-label
snd_labels['labels_list'] = snd_labels['primary_label'].apply(lambda x: [sp.strip() for sp in str(x).split(';') if sp.strip() in target_set])

# To enable StratifiedGroupKFold, we need a single representative class for each row to stratify on.
# For multi-label, we'll pick the rarest class or just the first class for stratification purposes.
# Alternatively, MultilabelStratifiedGroupKFold, but standard is fine for baseline if we pick the first label.
snd_labels['stratify_label'] = snd_labels['labels_list'].apply(lambda x: x[0] if len(x) > 0 else 'nocall')

train_df['labels_list'] = train_df['primary_label'].apply(lambda x: [x])
train_df['stratify_label'] = train_df['primary_label']

cols = ['filepath', 'start_sec', 'labels_list', 'stratify_label', 'group', 'is_soundscape']
full_df = pd.concat([train_df[cols], snd_labels[cols]], ignore_index=True)

# Build multi-label target matrix
targets = np.zeros((len(full_df), n_classes), dtype=np.float32)
for i, labels in enumerate(full_df['labels_list']):
    for sp in labels:
        targets[i, species_to_idx[sp]] = 1.0
        
full_df['target'] = list(targets)

# KFold
skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
full_df['fold'] = -1
for fold, (tr_idx, val_idx) in enumerate(skf.split(full_df, full_df['stratify_label'], groups=full_df['group'])):
    full_df.loc[val_idx, 'fold'] = fold

print("Folds distribution:")
print(full_df['fold'].value_counts())
""")

add_md("## 3. Dataset & Augmentations")

add_code("""class BirdDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Augmentations (Time & Freq Masking)
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        is_snd = row['is_soundscape']
        
        try:
            if is_snd:
                # Load specific segment
                start_frame = int(row['start_sec'] * SR)
                frames_to_read = int(SEGMENT_SEC * SR)
                y, sr_orig = torchaudio.load(filepath, frame_offset=start_frame, num_frames=frames_to_read)
            else:
                # Load train_audio
                y, sr_orig = torchaudio.load(filepath)
                # Random crop if training, else center crop/first 5 sec
                if y.shape[1] > SR * SEGMENT_SEC:
                    if self.is_train:
                        start = random.randint(0, y.shape[1] - SR * SEGMENT_SEC)
                    else:
                        start = 0
                    y = y[:, start:start + SR * SEGMENT_SEC]
                
            # Resample if needed
            if sr_orig != SR:
                y = torchaudio.functional.resample(y, orig_freq=sr_orig, new_freq=SR)
                
            # Convert to mono if stereo
            if y.shape[0] > 1:
                y = y.mean(dim=0, keepdim=True)
                
            # Pad if too short
            target_length = SR * SEGMENT_SEC
            if y.shape[1] < target_length:
                y = F.pad(y, (0, target_length - y.shape[1]))
                
        except Exception as e:
            y = torch.zeros(1, SR * SEGMENT_SEC)
            
        # Extract Mel Spec
        mel = self.mel_spec(y)
        mel = self.amplitude_to_db(mel)
        
        # Augmentations
        if self.is_train:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)
            
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Duplicate to 3 channels for CNN
        mel = mel.repeat(3, 1, 1)
        
        target = torch.tensor(row['target'], dtype=torch.float32)
        return mel, target
""")

add_md("## 4. Model Definition (EfficientNet-B2)")

add_code("""class BirdModel(nn.Module):
    def __init__(self, num_classes=n_classes, model_name='tf_efficientnet_b2_ns', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity() # Remove original classifier
        
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
""")

add_md("## 5. Training Loop & Validation")

add_code("""def train_fold(fold):
    print(f"\\n{'='*20} Fold {fold} {'='*20}")
    train_df_fold = full_df[full_df['fold'] != fold].reset_index(drop=True)
    val_df_fold = full_df[full_df['fold'] == fold].reset_index(drop=True)
    
    train_dataset = BirdDataset(train_df_fold, is_train=True)
    val_dataset = BirdDataset(val_df_fold, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = BirdModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_auc = 0
    oof_preds = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # AMP could be added here for speed
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(out).cpu().numpy())
                val_targets.append(y.cpu().numpy())
                
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # Macro ROC-AUC dropping classes with no true positives
        aucs = []
        for i in range(n_classes):
            if val_targets[:, i].sum() > 0:
                try:
                    auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass
        val_auc = np.mean(aucs) if len(aucs) > 0 else 0
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), OUT / f"fold_{fold}_best.pth")
            
    print(f"Best Fold {fold} AUC: {best_auc:.4f}")
    
    # Reload best model and get OOF preds
    model.load_state_dict(torch.load(OUT / f"fold_{fold}_best.pth"))
    model.eval()
    val_preds = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            val_preds.append(torch.sigmoid(out).cpu().numpy())
    val_preds = np.vstack(val_preds)
    
    return val_preds, val_df_fold

# Train all folds
oof_dfs = []
for fold in range(N_FOLDS):
    # Only training fold 0 for speed in demonstration/baseline.
    # To fully utilize, run all folds (remove the break).
    val_preds, val_df_fold = train_fold(fold)
    val_df_fold[[f'pred_{i}' for i in range(n_classes)]] = val_preds
    oof_dfs.append(val_df_fold)
    break # Remove this break to train all folds
""")

add_md("## 6. Threshold Optimization (Optional, disabled for now to just output probabilities)")
add_code("""# You can optimize class thresholds using oof_dfs
# For AUC, probabilities are sufficient. Thresholds matter for F1 or macro F1.
# Since eval metric is ROC-AUC, we leave probabilities as they are.
""")

add_md("## 7. Inference & Submission (Fixing the Test Fallback Bug)")

add_code("""test_files = sorted(TEST_SND.glob('*.ogg'))

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
    
    model = BirdModel(pretrained=False).to(device)
    # Load fold 0 for inference (if full ensemble, average across all loaded models)
    model.load_state_dict(torch.load(OUT / "fold_0_best.pth", map_location=device))
    model.eval()
    
    mel_spec = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB().to(device)
    
    all_rows = []
    
    for fp in tqdm(test_files, desc='Inferring test'):
        fname = fp.stem
        try:
            y_full, sr_orig = torchaudio.load(fp)
            if sr_orig != SR:
                y_full = torchaudio.functional.resample(y_full, orig_freq=sr_orig, new_freq=SR)
            if y_full.shape[0] > 1:
                y_full = y_full.mean(dim=0, keepdim=True)
        except Exception:
            # Handle broken files
            continue
            
        y_full = y_full.to(device)
        duration = y_full.shape[1] / SR
        n_segments = max(1, int(np.ceil(duration / SEGMENT_SEC))) # Bug fixed as per PLAN.md
        
        for seg_i in range(n_segments):
            start = seg_i * SEGMENT_SEC
            end = start + SEGMENT_SEC
            end_time = int(end)
            row_id = f'{fname}_{end_time}'
            
            start_frame = int(start * SR)
            end_frame = int(end * SR)
            segment = y_full[:, start_frame:end_frame]
            
            if segment.shape[1] < SR * SEGMENT_SEC:
                segment = F.pad(segment, (0, SR * SEGMENT_SEC - segment.shape[1]))
                
            mel = mel_spec(segment)
            mel = amplitude_to_db(mel)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            mel = mel.repeat(3, 1, 1).unsqueeze(0) # Batch size 1, 3 channels
            
            with torch.no_grad():
                out = model(mel)
                probs = torch.sigmoid(out)[0].cpu().numpy()
                
            all_rows.append((row_id, probs))
            
    if len(all_rows) == 0:
        print("No predictions generated. Using baseline probabilities.")
        probs = np.empty((0, n_classes))
        row_ids = []
    else:
        row_ids = [r[0] for r in all_rows]
        probs = np.stack([r[1] for r in all_rows], axis=0)

    sub = pd.DataFrame(probs, columns=SPECIES)
    sub.insert(0, 'row_id', row_ids)

    # Align to sample_submission
    sub = sample_sub[['row_id']].merge(sub, on='row_id', how='left')
    baseline_prob = 1.0 / n_classes
    sub[SPECIES] = sub[SPECIES].fillna(baseline_prob)

    submission_path = OUT / 'submission.csv'
    sub.to_csv(submission_path, index=False)
    print(f'submission.csv saved -> shape {sub.shape}')
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [{"sourceType": "competition", "sourceId": 129329, "databundleVersionId": 15996945}],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("/home/legionlinux/miniconda3/envs/torchenv/__INIT__/Kaggle/birdclef-2026/kaggle-notebooks/birdclef-2026-1.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook generated successfully!")
