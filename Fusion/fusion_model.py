# Fuse pooled RGB and pose features
import torch
from torch import nn
from torch.nn import functional as F


class FusionModel(nn.Module):
    def __init__(self, rgb_dim=2048, pose_dim=256, hidden_dim=256, num_classes=15):
        super().__init__()
        self.model_args = {
            "rgb_dim": rgb_dim, "pose_dim": pose_dim,
            "hidden_dim": hidden_dim, "num_classes": num_classes,
        }
        self.head = nn.Sequential(
            nn.Linear(rgb_dim + pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, rgb, pose):
        # Normalize each branch before concatenation
        rgb = F.normalize(rgb, dim=1)
        pose = F.normalize(pose, dim=1)
        features = torch.cat([rgb, pose], dim=1)
        return self.head(features)
