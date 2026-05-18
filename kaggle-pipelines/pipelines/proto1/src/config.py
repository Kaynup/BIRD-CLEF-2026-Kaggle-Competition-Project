import os
from pathlib import Path

class ConfigMeta(type):
    @property
    def OUTPUT_DIR(cls) -> Path:
        if getattr(cls, "_override_output_dir", None) is not None:
            return cls._override_output_dir
        
        # Dynamic fallback: find the latest existing run-X directory
        base_dir = cls.PIPELINE_ROOT / "outputs"
        if not base_dir.exists():
            return base_dir
        existing_runs = []
        for p in base_dir.iterdir():
            if p.is_dir() and p.name.startswith("run-"):
                try:
                    num = int(p.name.split("-")[1])
                    existing_runs.append(num)
                except ValueError:
                    pass
        if not existing_runs:
            return base_dir
        latest_num = max(existing_runs)
        return base_dir / f"run-{latest_num}"
        
    @OUTPUT_DIR.setter
    def OUTPUT_DIR(cls, value: Path):
        cls._override_output_dir = value

class Config(metaclass=ConfigMeta):
    # 1. Paths
    WORKSPACE_ROOT = Path("/home/legionlinux/miniconda3/envs/torchenv/__INIT__/Kaggle/birdclef-2026")
    PIPELINE_ROOT = WORKSPACE_ROOT / "kaggle-pipelines" / "pipelines" / "proto1"
    
    DATA_RAW = WORKSPACE_ROOT / "data" / "raw"
    TRAIN_AUDIO_DIR = DATA_RAW / "train_audio"
    TRAIN_SOUNDSCAPES_DIR = DATA_RAW / "train_soundscapes"
    
    TRAIN_CSV = DATA_RAW / "train.csv"
    SOUNDSCAPES_CSV = DATA_RAW / "train_soundscapes_labels.csv"
    TAXONOMY_CSV = DATA_RAW / "taxonomy.csv"
    SAMPLE_SUB_CSV = DATA_RAW / "sample_submission.csv"

    # 2. Audio & Spectrogram Parameters
    SAMPLE_RATE = 32000
    DURATION_SECONDS = 5.0
    CHUNK_LENGTH = int(SAMPLE_RATE * DURATION_SECONDS)  # 160,000 samples
    
    # Mel Spectrogram Settings
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    F_MIN = 50
    F_MAX = 16000
    
    # 3. Model Hyperparameters
    BACKBONE_NAME = "resnet34"  # Available in torchvision.models
    NUM_CLASSES = 234
    
    # Spatial MLP
    L_FREQ = 6  # Number of sinusoidal positional frequencies
    SPATIAL_MLP_DIMS = [36, 128, 64, NUM_CLASSES]  # input_dim = 3 coord * 2 (sin/cos) * L = 36
    
    # Relational Species Head (Query2Label)
    SPECIES_EMB_DIM = 256  # Dimension of species query tokens
    BACKBONE_OUT_DIM = 512  # ResNet34 average pooling out size is 512
    ATTENTION_HEADS = 4
    ATTENTION_DEPTH = 1
    
    # Pantanal Bounding Box Constants
    PANTANAL_LAT_CENTER = -19.05
    PANTANAL_LON_CENTER = -56.75
    
    # 4. Training Hyperparameters
    NUM_FOLDS = 5
    EPOCHS = 15
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    COSINE_LR_MIN = 1e-6
    
    # Regularization
    SPATIAL_DROPOUT_PROB = 0.5  # Coordinate dropout probability
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.5
    
    # SPEC Augment Settings
    FREQ_MASK_PARAM = 10
    TIME_MASK_PARAM = 20
    
    # 5. Checkpoint Gating
    GENERALIZATION_PENALTY_GAMMA = 0.5
