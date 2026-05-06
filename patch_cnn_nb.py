import json

path = "/home/legionlinux/miniconda3/envs/torchenv/__INIT__/Kaggle/birdclef-2026/kaggle-notebooks/birdclef-2026-cnn.ipynb"

with open(path, "r") as f:
    nb = json.load(f)

# Find the inference cell
inf_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "test_files = sorted(TEST_SND.glob('*.ogg'))" in "".join(cell["source"]):
        inf_idx = i
        break

if inf_idx == -1:
    print("Inference cell not found!")
    exit(1)

new_code = """import concurrent.futures
import soundfile as sf
import math

test_files = sorted(TEST_SND.glob('*.ogg'))
IS_DRY_RUN = len(test_files) == 0

if IS_DRY_RUN:
    print("No hidden test files found. Dry-run on 10 train soundscapes...")
    test_files = sorted(TRAIN_SND.glob('*.ogg'))[:10]
else:
    print(f'Hidden test soundscapes found: {len(test_files)}')

model = BirdModel(pretrained=False).to(device)
try:
    model.load_state_dict(torch.load(OUT / "fold_0_best.pth", map_location=device))
    print("Loaded trained fold 0 successfully.")
except Exception as e:
    print(f"WARNING: Could not load trained fold model. {e}")
    
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

def read_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    # Ensure it's not empty and right sample rate is assumed (Kaggle tests are 32k)
    return y

BATCH_FILES = 8 # Process 8 files (8 * 12 = 96 windows) at a time
N_WINDOWS_PER_FILE = 12

all_rows = []
probs_list = []

# Using ThreadPoolExecutor for background file reading
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_exec:
    for i in tqdm(range(0, len(test_files), BATCH_FILES), desc="Inferring test batches"):
        batch_paths = test_files[i:i+BATCH_FILES]
        
        # Multithreaded read
        future_audio = [io_exec.submit(read_audio, p) for p in batch_paths]
        batch_audio = [f.result() for f in future_audio]
        
        batch_mels = []
        batch_row_ids = []
        
        for bi, path in enumerate(batch_paths):
            fname = path.stem
            y = batch_audio[bi]
            
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            duration = y_tensor.shape[1] / SR
            n_segments = max(1, int(np.ceil(duration / SEGMENT_SEC)))
            
            for seg_i in range(n_segments):
                start = seg_i * SEGMENT_SEC
                end = start + SEGMENT_SEC
                end_time = int(end)
                row_id = f'{fname}_{end_time}'
                
                start_frame = int(start * SR)
                end_frame = int(end * SR)
                segment = y_tensor[:, start_frame:end_frame]
                
                if segment.shape[1] < SR * SEGMENT_SEC:
                    segment = F.pad(segment, (0, SR * SEGMENT_SEC - segment.shape[1]))
                    
                mel = mel_spec(segment)
                mel = amplitude_to_db(mel)
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)
                mel = mel.repeat(3, 1, 1) # Shape: (3, MELS, TIME)
                
                batch_mels.append(mel)
                batch_row_ids.append(row_id)
                
        if not batch_mels:
            continue
            
        # Batch inference
        batch_mels_tensor = torch.stack(batch_mels).to(device) # Shape: (B, 3, MELS, TIME)
        
        with torch.no_grad():
            # If batch is too large, you could split it here, but 8*12=96 is fine for B2
            out = model(batch_mels_tensor)
            batch_probs = torch.sigmoid(out).cpu().numpy()
            
        all_rows.extend(batch_row_ids)
        probs_list.append(batch_probs)

if len(all_rows) == 0:
    print("No predictions generated. Using baseline probabilities.")
    probs = np.empty((0, n_classes))
    row_ids = []
else:
    probs = np.vstack(probs_list)
    row_ids = all_rows

sub = pd.DataFrame(probs, columns=SPECIES)
sub.insert(0, 'row_id', row_ids)

# We must output exactly the rows in sample_submission, and handle dry_run gracefully
if IS_DRY_RUN:
    print("Dry-run: formatting submission to match sample_submission...")
    sample_pub = pd.read_csv(BASE / 'sample_submission.csv')
    mean_pred = sub[SPECIES].mean(axis=0).fillna(1.0/n_classes).to_dict()
    sub = sample_pub.copy()
    for sp in SPECIES:
        sub[sp] = mean_pred[sp]
else:
    sub = sample_sub[['row_id']].merge(sub, on='row_id', how='left')
    baseline_prob = 1.0 / n_classes
    sub[SPECIES] = sub[SPECIES].fillna(baseline_prob)

submission_path = OUT / 'submission.csv'
sub.to_csv(submission_path, index=False)
print(f'submission.csv saved -> shape {sub.shape}')
"""

# Format back to list of strings
nb["cells"][inf_idx]["source"] = [line + "\n" for line in new_code.split("\n")]
# Remove the last trailing newline from the last line to be clean
if len(nb["cells"][inf_idx]["source"]) > 0:
    nb["cells"][inf_idx]["source"][-1] = nb["cells"][inf_idx]["source"][-1][:-1]

with open(path, "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook inference cell patched successfully!")
