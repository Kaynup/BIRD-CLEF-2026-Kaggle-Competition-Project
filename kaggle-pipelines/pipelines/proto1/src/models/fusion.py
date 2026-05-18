import torch
import torch.nn as nn

class LogitGatedFusion(nn.Module):
    """
    Logit-Level Learned Gating Fusion module.
    Blends audio-derived logits and coordinate-derived spatial logits:
    l_final = l_audio + alpha * l_spatial
    where alpha is a learnable scale parameter initialized to 1.0.
    """
    def __init__(self):
        super().__init__()
        # 1. Initialize learnable scale alpha
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
    def forward(self, logits_audio: torch.Tensor, logits_spatial: torch.Tensor) -> torch.Tensor:
        # Enforce non-negativity on alpha to keep it a supportive prior
        alpha_gated = torch.clamp(self.alpha, min=0.0)
        
        # Fuse logits
        logits_final = logits_audio + alpha_gated * logits_spatial
        
        return logits_final
