import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np

class SpecAugment(nn.Module):
    """
    Applies SpecAugment (Frequency and Time masking) on Mel-Spectrograms.
    Prevents the model from overfitting to specific hums, noise bands, or alignments.
    """
    def __init__(self, freq_mask_param: int = 10, time_mask_param: int = 20):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [..., freq, time]
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x

class Mixup:
    """
    Performs multi-label batch-level Mixup augmentation.
    Blends raw spectrograms and their target labels:
    x_mix = lambda * x1 + (1 - lambda) * x2
    y_mix = lambda * y1 + (1 - lambda) * y2
    """
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if np.random.rand() > self.prob:
            return x, y
            
        batch_size = x.size(0)
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Shuffle indices
        indices = torch.randperm(batch_size, device=x.device)
        
        # Mix inputs
        x_mixed = lam * x + (1.0 - lam) * x[indices]
        
        # Mix multi-label target vectors
        y_mixed = lam * y + (1.0 - lam) * y[indices]
        
        return x_mixed, y_mixed

def normalize_spectrogram(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a log-mel spectrogram to match ImageNet standard 3-channel layout.
    Expects x of shape [freq, time]. Output shape [3, freq, time].
    """
    # 1. Normalize between 0 and 1
    min_val = x.min()
    max_val = x.max()
    if max_val - min_val > 1e-5:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = torch.zeros_like(x)
        
    # 2. Duplicate to 3 channels (required by ImageNet pre-trained vision CNNs)
    x = x.unsqueeze(0).repeat(3, 1, 1)
    
    # 3. Apply ImageNet standard normalization (mean & std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)
    
    x = (x - mean) / std
    return x
