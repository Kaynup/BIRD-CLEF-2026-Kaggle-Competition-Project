import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config

class AudioVisionBackbone(nn.Module):
    """
    CNN / Vision Backbone wrapping torchvision models.
    Takes 3-channel log-mel spectrograms [3, N_MELS, TIME]
    and extracts high-level spatial-temporal feature maps.
    """
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        
        # Load backbone from torchvision.models
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            # Remove original avgpool and fc layers
            self.features = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            )
            self.out_channels = 512
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.features = self.backbone.features
            self.out_channels = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} is not supported. Use resnet34 or efficientnet_b0.")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, 3, N_MELS, TIME]
        x = self.features(x)
        # Output shape: [Batch, out_channels, H_feat, W_feat]
        return x
