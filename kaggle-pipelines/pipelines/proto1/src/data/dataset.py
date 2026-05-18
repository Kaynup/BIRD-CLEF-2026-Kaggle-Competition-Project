import torch
import torchaudio
import numpy as np
import pandas as pd
import ast
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple
from src.config import Config
from src.data.transforms import normalize_spectrogram
from src.utils.spatial import lat_lon_to_cartesian, get_sinusoidal_positional_encoding

def parse_time_to_seconds(time_str: str) -> float:
    """Parses a time string of format HH:MM:SS to float seconds."""
    parts = list(map(float, time_str.split(':')))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]

class BirdCLEFDataset(Dataset):
    """
    Unified high-performance PyTorch Dataset handling:
    1. Short audio clips (train_audio) with single primary + background secondary labels.
    2. Continuous soundscapes (train_soundscapes) with multi-label expert 5s segments.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        species_to_idx: Dict[str, int], 
        is_train: bool = True,
        spec_augment = None
    ):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.is_train = is_train
        self.spec_augment = spec_augment
        
        # Instantiate on-the-fly Spectrogram transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX
        )
        
    def __len__(self) -> int:
        return len(self.df)
        
    def _load_audio_segment(self, path: Path, start_sec: float = 0.0, duration_sec: float = 5.0) -> torch.Tensor:
        """
        Efficiently loads a specific slice of audio from disk using soundfile.
        Bypasses deprecated torchaudio loading backends.
        """
        try:
            info = sf.info(path)
            sr = info.samplerate
            
            frame_offset = int(start_sec * sr)
            num_frames = int(duration_sec * sr)
            
            # Load specific slice natively via soundfile
            data, sample_rate = sf.read(path, start=frame_offset, frames=num_frames, dtype='float32')
            
            # Convert to PyTorch tensor of shape (channels, frames)
            if data.ndim == 1:
                waveform = torch.tensor(data).unsqueeze(0)
            else:
                waveform = torch.tensor(data).T
                
            # Resample to 32kHz if necessary using torchaudio's clean tensor resampler
            if sample_rate != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLE_RATE)
                waveform = resampler(waveform)
                
        except Exception as e:
            # Fallback for corrupted/unreadable files (return quiet vector)
            waveform = torch.zeros((1, Config.CHUNK_LENGTH))
            
        # Standardize channels (mono conversion)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or Crop to exact chunk length
        if waveform.size(1) < Config.CHUNK_LENGTH:
            pad_len = Config.CHUNK_LENGTH - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.size(1) > Config.CHUNK_LENGTH:
            if self.is_train:
                # Random crop during training
                max_start = waveform.size(1) - Config.CHUNK_LENGTH
                start_frame = np.random.randint(0, max_start)
                waveform = waveform[:, start_frame:start_frame + Config.CHUNK_LENGTH]
            else:
                # Center crop during validation
                start_frame = (waveform.size(1) - Config.CHUNK_LENGTH) // 2
                waveform = waveform[:, start_frame:start_frame + Config.CHUNK_LENGTH]
                
        return waveform.squeeze(0)  # Shape: [CHUNK_LENGTH]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # 1. Determine type of data and load audio
        if 'start' in row and not pd.isna(row['start']):
            # Expert-labeled Soundscape segment
            audio_path = Config.TRAIN_SOUNDSCAPES_DIR / row['filename']
            start_sec = parse_time_to_seconds(row['start'])
            waveform = self._load_audio_segment(audio_path, start_sec=start_sec, duration_sec=5.0)
            
            # Coordinates (Soundscapes are native to Pantanal)
            lat = torch.tensor(Config.PANTANAL_LAT_CENTER, dtype=torch.float32)
            lon = torch.tensor(Config.PANTANAL_LON_CENTER, dtype=torch.float32)
            
            # Target Label Mapping
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            labels = str(row['primary_label']).split(';')
            for label in labels:
                label = label.strip()
                if label in self.species_to_idx:
                    target[self.species_to_idx[label]] = 1.0
        else:
            # Short Audio Clip from train_audio
            audio_path = Config.TRAIN_AUDIO_DIR / row['filename']
            # Load entire clip and crop 5s randomly/center
            waveform = self._load_audio_segment(audio_path, start_sec=0.0, duration_sec=60.0) # Load up to 60s max to save mem
            
            # Coordinates
            lat = torch.tensor(row['latitude'], dtype=torch.float32)
            lon = torch.tensor(row['longitude'], dtype=torch.float32)
            
            # Target Label Mapping
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            primary = str(row['primary_label']).strip()
            if primary in self.species_to_idx:
                target[self.species_to_idx[primary]] = 1.0
                
            # Parse secondary background labels (weighted at 0.5)
            try:
                secondaries = ast.literal_eval(row['secondary_labels'])
                for label in secondaries:
                    label = label.strip()
                    if label in self.species_to_idx:
                        target[self.species_to_idx[label]] = 0.5
            except:
                pass
                
        # 2. Extract Spectrogram
        mel_spec = self.mel_transform(waveform)  # Shape [N_MELS, TIME]
        
        # 3. Stabilized Log Scaling
        log_mel = torch.log(mel_spec + 1e-6)
        
        # 4. Augmentation (SpecAugment)
        if self.is_train and self.spec_augment is not None:
            log_mel = self.spec_augment(log_mel)
            
        # 5. Normalization
        spectrogram = normalize_spectrogram(log_mel)
        
        # 6. Coordinate Processing
        # Map to Cartesian 3D unit sphere
        cartesian = lat_lon_to_cartesian(lat, lon)  # Shape [3]
        # Get sinusoidal positional encodings
        pe_coords = get_sinusoidal_positional_encoding(cartesian.unsqueeze(0), L=Config.L_FREQ).squeeze(0)  # Shape [36]
        
        return spectrogram, pe_coords, target
