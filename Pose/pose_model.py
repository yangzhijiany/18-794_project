# Pose-only model: ST-GCN on the 13-joint Penn Action skeleton
#
# ST-GCN follows the reference implementation
# https://github.com/yysijie/st-gcn (net/st_gcn.py, net/utils/graph.py)
# Paper: Yan et al., "Spatial Temporal Graph Convolutional Networks for
# Skeleton-Based Action Recognition", AAAI 2018.
#
# Penn has no neck or pelvis joint, so the "spatial" partition measures the
# distance to the body centre as the hop distance to the nearest hip.

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .pose_transform import (
    DEFAULT_FALLBACK_K, HIPS, NUM_JOINTS, augment_pose, normalize_pose,
)

# 0 head; odd = right, even = left (see pose_transform.py)
PENN_EDGES = [
    (0, 1), (0, 2), (1, 2),            # head - shoulders, shoulder - shoulder
    (1, 3), (3, 5), (2, 4), (4, 6),    # arms
    (1, 7), (2, 8), (7, 8),            # torso sides, hip - hip
    (7, 9), (9, 11), (8, 10), (10, 12),  # legs
]


def penn_hop_distance(num_nodes=NUM_JOINTS, edges=PENN_EDGES):
    hop = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(hop, 0)
    for i, j in edges:
        hop[i, j] = hop[j, i] = 1

    # Floyd-Warshall on an unweighted graph
    for k in range(num_nodes):
        hop = np.minimum(hop, hop[:, k:k + 1] + hop[k:k + 1, :])
    return hop


def penn_spatial_adjacency():
    """[3, 13, 13] partitions of the ST-GCN spatial strategy (max_hop = 1)."""
    hop = penn_hop_distance()
    centre_dist = hop[:, HIPS].min(axis=1)

    adjacency = (hop <= 1).astype(np.float64)
    degree = adjacency.sum(axis=0)
    normalized = adjacency / degree[None, :]   # A @ D^-1, as in the reference

    root = np.zeros_like(normalized)
    close = np.zeros_like(normalized)
    further = np.zeros_like(normalized)
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if hop[j, i] == 0:
                root[j, i] = normalized[j, i]
            elif hop[j, i] == 1:
                if centre_dist[j] == centre_dist[i]:
                    close[j, i] = normalized[j, i]
                elif centre_dist[j] > centre_dist[i]:
                    close[j, i] = normalized[j, i]
                else:
                    further[j, i] = normalized[j, i]

    return torch.tensor(np.stack([root, close, further]), dtype=torch.float32)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions):
        super().__init__()
        self.num_partitions = num_partitions
        self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)

    def forward(self, x, adjacency):
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.num_partitions, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, adjacency)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions, stride=1,
                 residual=True, temporal_kernel=9, dropout=0.0):
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, num_partitions)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), (stride, 1), (padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacency):
        out = self.tcn(self.gcn(x, adjacency))
        if self.residual is not None:
            out = out + self.residual(x)
        return self.relu(out)


class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=15, base_channels=64,
                 temporal_kernel=9, dropout=0.5, block_dropout=0.1):
        super().__init__()
        self.register_buffer("adjacency", penn_spatial_adjacency())
        k = self.adjacency.shape[0]
        c = base_channels

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        layout = [
            (in_channels, c, 1, False), (c, c, 1, True), (c, c, 1, True), (c, c, 1, True),
            (c, 2 * c, 2, True), (2 * c, 2 * c, 1, True), (2 * c, 2 * c, 1, True),
            (2 * c, 4 * c, 2, True), (4 * c, 4 * c, 1, True), (4 * c, 4 * c, 1, True),
        ]
        self.blocks = nn.ModuleList([
            STGCNBlock(cin, cout, k, stride, residual, temporal_kernel,
                       0.0 if i == 0 else block_dropout)
            for i, (cin, cout, stride, residual) in enumerate(layout)
        ])
        # Learnable edge importance, one mask per block
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones_like(self.adjacency)) for _ in layout
        ])

        self.feature_dim = 4 * c
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def features(self, x):
        # x [N, C, T, V] -> pooled features [N, 4c]
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x = self.data_bn(x).view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.adjacency * importance)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))


class PoseModel(nn.Module):
    """Takes the raw Dataset output; normalization happens inside forward().

    forward(batch["pose"], batch["joint_visibility"]) -> logits [B, num_classes]
    extract_features(...) -> [B, feature_dim] (pooled, before dropout/fc, not L2-normalized)
    """

    num_frames = 16
    num_joints = NUM_JOINTS
    input_modes = ("xyv", "xy", "v")

    def __init__(self, num_classes=15, norm_mode="bbox", fallback_k=DEFAULT_FALLBACK_K,
                 base_channels=64, temporal_kernel=9, dropout=0.5, block_dropout=0.1,
                 input_mode="xyv"):
        super().__init__()
        if input_mode not in self.input_modes:
            raise ValueError(f"input_mode must be one of {self.input_modes}")

        self.model_args = {
            "num_classes": num_classes, "norm_mode": norm_mode, "fallback_k": float(fallback_k),
            "base_channels": base_channels, "temporal_kernel": temporal_kernel,
            "dropout": dropout, "block_dropout": block_dropout, "input_mode": input_mode,
        }
        self.num_classes = num_classes
        self.norm_mode = norm_mode
        self.fallback_k = float(fallback_k)
        self.input_mode = input_mode
        self.augment = False   # training script turns this on; never active in eval()

        self.backbone = STGCN(3, num_classes, base_channels, temporal_kernel, dropout, block_dropout)
        self.feature_dim = self.backbone.feature_dim

    def preprocess(self, pose, visibility):
        expected = (self.num_frames, self.num_joints, 2)
        if pose.ndim != 4 or tuple(pose.shape[1:]) != expected:
            raise ValueError(f"Expected pose [B, 16, 13, 2], got {tuple(pose.shape)}")
        return normalize_pose(pose, visibility, self.norm_mode, self.fallback_k)

    def _select_channels(self, x):
        # Used only by the input ablations (xy without v, or v alone)
        if self.input_mode == "xy":
            x = torch.cat([x[:, :2], torch.zeros_like(x[:, 2:])], dim=1)
        elif self.input_mode == "v":
            x = torch.cat([torch.zeros_like(x[:, :2]), x[:, 2:]], dim=1)
        return x

    def prepare(self, pose, visibility):
        x = self.preprocess(pose, visibility)
        if self.training and self.augment:
            x = augment_pose(x)
        return self._select_channels(x)

    def forward(self, pose, visibility):
        return self.backbone(self.prepare(pose, visibility))

    def extract_features(self, pose, visibility):
        return self.backbone.features(self.prepare(pose, visibility))

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
        model = cls(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model"])
        return model
