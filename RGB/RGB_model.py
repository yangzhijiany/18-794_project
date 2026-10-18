# 15-class Penn Action model, using the SS-v1 feature weights
# The dropout -> per-frame classifier -> temporal mean follows
# https://github.com/swathikirans/GSF/blob/9d4d80e33dc81f8de882b583b7062606411cb696/models.py
# Paper: https://doi.org/10.1109/TPAMI.2023.3268134

from pathlib import Path
import torch
from torch import nn
from .resnetGSFModels import ResNet50GSF


class RGBModel(nn.Module):
    num_frames = 16
    feature_dim = 2048

    def __init__(self, num_classes=15):
        super().__init__()
        self.num_classes = num_classes

        # Keep these attribute names identical to the author's checkpoint
        self.base_model = ResNet50GSF()
        self.base_model.fc = nn.Dropout(0.5)
        self.new_fc = nn.Linear(self.feature_dim, num_classes)

        nn.init.normal_(self.new_fc.weight, mean=0, std=0.001)
        nn.init.zeros_(self.new_fc.bias)

        self.backbone_frozen = False

    def frame_features(self, rgb):
        expected = (self.num_frames, 3, 224, 224)
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"Expected [B, 16, 3, 224, 224], got {tuple(rgb.shape)}")
        
        batch_size = rgb.shape[0]
        frames = rgb.reshape(batch_size * self.num_frames, 3, 224, 224)
        features = self.base_model(frames)

        # restore the dimensions
        result = features.reshape(batch_size, self.num_frames, self.feature_dim)
        return result

    def forward(self, rgb):
        features = self.frame_features(rgb)
        frame_logits = self.new_fc(features)

        result = frame_logits.mean(dim=1)
        return result  

    def extract_features(self, rgb):
        return self.frame_features(rgb).mean(dim=1)

    def freeze_backbone(self, frozen=True):
        self.backbone_frozen = frozen
        self.base_model.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            # Freezing BN must not change during training
            self.base_model.eval()
            self.base_model.fc.train(mode)
        return self


def load_pretrained(model, path):
    # Use backbone and leave our new classifier untouched
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    prefix = "module.base_model."
    backbone_weights = {}
    
    for name, weight in checkpoint["model_state_dict"].items():
        if name.startswith(prefix):
            layer_name = name[len(prefix):]
            backbone_weights[layer_name] = weight

    model.base_model.load_state_dict(backbone_weights, strict=True)