import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.config import Config
from src.utils.spatial import lat_lon_to_cartesian, get_sinusoidal_positional_encoding
from src.data.transforms import normalize_spectrogram
from src.models import MultiModalBirdModel

def main():
    print("Initializing submission inference pipeline...")
    
    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Inference Device: {device}")
    
    # 2. Determine Test Directory (Fallback to Train Soundscapes for local validation)
    # Kaggle test path is typically /kaggle/input/birdclef-2026/test_soundscapes/
    kaggle_test_dir = Path("/kaggle/input/birdclef-2026/test_soundscapes")
    if kaggle_test_dir.exists():
        test_dir = kaggle_test_dir
        is_real_test = True
        print(f"Real test directory found: {test_dir}")
    else:
        # Fallback to local train soundscapes for testing
        test_dir = Config.TRAIN_SOUNDSCAPES_DIR
        is_real_test = False
        print(f"Test directory not found. Falling back to local validation soundscapes: {test_dir}")
        
    # 3. Load Target Species Columns
    sub_df = pd.read_csv(Config.SAMPLE_SUB_CSV)
    target_species = list(sub_df.columns)[1:]  # Exclude row_id
    species_to_idx = {species: idx for idx, species in enumerate(target_species)}
    
    # 4. Load Pre-trained Fold Weights
    model = MultiModalBirdModel(
        backbone_name=Config.BACKBONE_NAME,
        pretrained=False,  # Weights are loaded from checkpoint
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    checkpoint_path = Config.OUTPUT_DIR / "best_model_fold_0.pth"
    if checkpoint_path.exists():
        print(f"Loading best fold model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No pre-trained weights found! Running inference with random baseline initializations.")
        
    model.eval()
    
    # 5. Pre-compute Spherical Cartesian coordinate positional encodings
    # All continuous soundscapes are native to the Pantanal recording deployment box
    lat = torch.tensor(Config.PANTANAL_LAT_CENTER, dtype=torch.float32)
    lon = torch.tensor(Config.PANTANAL_LON_CENTER, dtype=torch.float32)
    
    cartesian = lat_lon_to_cartesian(lat, lon)
    pe_coords = get_sinusoidal_positional_encoding(cartesian.unsqueeze(0), L=Config.L_FREQ).to(device) # Shape: [1, 36]
    
    # Instantiate Mel Spectrogram extractor
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX
    ).to(device)
    
    # 6. List audio files to process
    audio_files = sorted(list(test_dir.glob("*.ogg")))
    
    # If in local test fallback mode, process a subset of 3 files to save time
    if not is_real_test:
        audio_files = audio_files[:3]
        print(f"Fallback Mode: Processing a subset of {len(audio_files)} files to verify correctness.")
        
    submission_rows = []
    
    # 7. Process each soundscape
    for path in tqdm(audio_files, desc="Inference"):
        filename_stem = path.stem
        
        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            total_frames = info.num_frames
            duration = total_frames / sr
        except Exception as e:
            # Fallback if file info fails
            sr = Config.SAMPLE_RATE
            duration = 60.0
            
        # We chunk each 60-second file into 12 segments of 5 seconds
        num_segments = int(duration // Config.DURATION_SECONDS)
        
        for segment_idx in range(num_segments):
            end_seconds = int((segment_idx + 1) * Config.DURATION_SECONDS)
            row_id = f"{filename_stem}_{end_seconds}"
            
            # Load specific 5s audio chunk
            frame_offset = int(segment_idx * Config.DURATION_SECONDS * sr)
            num_frames = int(Config.DURATION_SECONDS * sr)
            
            try:
                waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
                
                # Resample to 32kHz if necessary
                if sample_rate != Config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLE_RATE).to(device)
                    waveform = resampler(waveform.to(device)).cpu()
            except Exception as e:
                # Quiet fallback tensor if slice loading fails
                waveform = torch.zeros((1, Config.CHUNK_LENGTH))
                
            # Mono conversion
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Pad to exact chunk length
            if waveform.size(1) < Config.CHUNK_LENGTH:
                pad_len = Config.CHUNK_LENGTH - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif waveform.size(1) > Config.CHUNK_LENGTH:
                waveform = waveform[:, :Config.CHUNK_LENGTH]
                
            # Move waveform to device and extract mel spectrogram
            waveform = waveform.to(device).squeeze(0)  # Shape [CHUNK_LENGTH]
            
            with torch.no_grad():
                # On-the-fly Mel-spectrogram
                mel_spec = mel_transform(waveform)  # Shape [128, TIME]
                log_mel = torch.log(mel_spec + 1e-6)
                
                # Normalization
                spectrogram = normalize_spectrogram(log_mel).unsqueeze(0)  # Shape [1, 3, 128, TIME]
                
                # Forward pass with coordinate dropout disabled
                logits = model(spectrogram, pe_coords, apply_dropout=False)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # Shape [234]
                
            # Construct row
            row = {'row_id': row_id}
            for species, idx in species_to_idx.items():
                row[species] = float(probs[idx])
                
            submission_rows.append(row)
            
    # 8. Create final submission file
    submission_df = pd.DataFrame(submission_rows)
    
    # Save submission file to base workspace directory
    output_sub_path = Config.WORKSPACE_ROOT / "submission.csv"
    submission_df.to_csv(output_sub_path, index=False)
    
    print(f"Inference Completed! Submission file successfully saved to: {output_sub_path}")
    print(submission_df.head(3))

if __name__ == "__main__":
    main()
