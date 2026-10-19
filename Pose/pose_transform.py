# Pose preprocessing for Penn Action (13 joints)
#
# Joint order, checked on the labels and on frames:
#   0 head, then (right, left) pairs: 1/2 shoulder, 3/4 elbow, 5/6 wrist,
#   7/8 hip, 9/10 knee, 11/12 ankle.
# Odd index = the person's RIGHT side (front view: index 1 is left of index 2
# in the image in 96.9% of frames; back view: reversed in 81.7%).
#
# visibility = 0 coordinates are not usable: about a third are a sentinel near
# (0, 0) and most of the rest are frozen copies of the previous frame. So all
# statistics use visible joints only, and invisible joints become 0 after
# normalization, with visibility kept as a third input channel.
#
# Every statistic is computed from the T sampled frames of the clip, which all
# lie inside the observed prefix, so nothing after the prefix is used.

import numpy as np
import torch

JOINT_NAMES = [
    "head",
    "r_shoulder", "l_shoulder",
    "r_elbow", "l_elbow",
    "r_wrist", "l_wrist",
    "r_hip", "l_hip",
    "r_knee", "l_knee",
    "r_ankle", "l_ankle",
]
NUM_JOINTS = len(JOINT_NAMES)
SHOULDERS = [1, 2]
HIPS = [7, 8]
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
NORM_MODES = ("torso", "bbox")

# Median of (torso length / diagonal of the visible-joint box) over the
# training split, used to put the fallback scale on the torso scale.
# Recomputed by calibrate_fallback_k() at the start of training.
DEFAULT_FALLBACK_K = 0.259


def _per_frame_mean(pose, mask, joints):
    # Mean of the visible joints in `joints`, per frame -> [B, T, 2], [B, T]
    m = mask[:, :, joints].float()
    total = (pose[:, :, joints] * m.unsqueeze(-1)).sum(dim=2)
    count = m.sum(dim=2)
    return total / count.clamp(min=1).unsqueeze(-1), count > 0


def _visible_box(pose, mask):
    # Box around all visible joints of the clip -> centre [B, 2], diagonal [B]
    m = mask.unsqueeze(-1)
    lo = torch.where(m, pose, torch.full_like(pose, float("inf"))).amin(dim=(1, 2))
    hi = torch.where(m, pose, torch.full_like(pose, float("-inf"))).amax(dim=(1, 2))

    has_any = mask.flatten(1).any(dim=1, keepdim=True)
    lo = torch.where(has_any, lo, torch.zeros_like(lo))
    hi = torch.where(has_any, hi, torch.zeros_like(hi))
    return (lo + hi) / 2, (hi - lo).norm(dim=-1)


def _masked_median(values, valid):
    # Median over dim 1 of values[valid] (mean of the two middle values) -> [B]
    filled = torch.where(valid, values, torch.full_like(values, float("inf")))
    ordered = filled.sort(dim=1).values
    count = valid.sum(dim=1)
    low = ((count - 1).clamp(min=0) // 2).unsqueeze(1)
    high = (count // 2).clamp(max=values.shape[1] - 1).unsqueeze(1)
    median = (ordered.gather(1, low) + ordered.gather(1, high)).squeeze(1) / 2
    return torch.where(count > 0, median, torch.zeros_like(median))


def normalize_pose(pose, visibility, mode="torso", fallback_k=DEFAULT_FALLBACK_K):
    """pose [B, T, 13, 2] raw pixels of the sampled frames, visibility [B, T, 13].

    Returns [B, 3, T, 13] with channels (x, y, v).
    torso: centre = mean hip position over the clip, scale = median torso length;
           fallback to the visible-joint box when hips / torso are not visible.
    bbox:  centre = centre of the visible-joint box, scale = k * its diagonal.
    """
    if mode not in NORM_MODES:
        raise ValueError(f"mode must be one of {NORM_MODES}")
    if pose.ndim != 4 or pose.shape[-2:] != (NUM_JOINTS, 2):
        raise ValueError(f"Expected pose [B, T, 13, 2], got {tuple(pose.shape)}")

    pose = pose.float()
    mask = visibility > 0.5
    box_centre, box_diag = _visible_box(pose, mask)
    box_scale = box_diag * float(fallback_k)

    if mode == "torso":
        hip, hip_frames = _per_frame_mean(pose, mask, HIPS)
        shoulder, shoulder_frames = _per_frame_mean(pose, mask, SHOULDERS)

        hip_weight = hip_frames.float().unsqueeze(-1)
        hip_centre = (hip * hip_weight).sum(dim=1) / hip_weight.sum(dim=1).clamp(min=1)
        has_hip = hip_frames.any(dim=1, keepdim=True)
        centre = torch.where(has_hip, hip_centre, box_centre)

        torso_frames = hip_frames & shoulder_frames
        torso = _masked_median((shoulder - hip).norm(dim=-1), torso_frames)
        scale = torch.where(torso_frames.any(dim=1), torso, box_scale)
    else:
        centre = box_centre
        scale = box_scale

    # A single visible joint (or none) gives no scale: keep pixels / 1
    scale = torch.where(scale > 1e-3, scale, torch.ones_like(scale))

    coords = (pose - centre[:, None, None, :]) / scale[:, None, None, None]
    coords = coords * mask.unsqueeze(-1)

    x = torch.cat([coords, mask.float().unsqueeze(-1)], dim=-1)
    return x.permute(0, 3, 1, 2).contiguous()


def flip_pose(x):
    # Horizontal flip of normalized input [B, 3, T, V]: swap left/right, negate x
    x = x[..., FLIP_INDEX].clone()
    x[:, 0] = -x[:, 0]
    return x


def augment_pose(x, flip_prob=0.5, max_degrees=15.0, scale_range=(0.9, 1.1)):
    # Training-only augmentation on normalized input [B, 3, T, V]
    batch = x.shape[0]
    flip = torch.rand(batch, device=x.device) < flip_prob
    x = torch.where(flip[:, None, None, None], flip_pose(x), x)

    angle = torch.deg2rad((torch.rand(batch, device=x.device) * 2 - 1) * max_degrees)
    scale = torch.empty(batch, device=x.device).uniform_(*scale_range)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale

    px, py = x[:, 0], x[:, 1]
    rx = cos[:, None, None] * px - sin[:, None, None] * py
    ry = sin[:, None, None] * px + cos[:, None, None] * py
    # Rotation about the origin keeps invisible joints at 0
    return torch.stack([rx, ry, x[:, 2]], dim=1)


def calibrate_fallback_k(samples):
    # samples: PennDataset.samples of the TRAINING split (dataset-level constant)
    ratios = []
    for sample in samples:
        pose, mask = sample["pose"], sample["visibility"] > 0.5
        if mask.sum() < 3:
            continue

        def frame_mean(joints):
            m = mask[:, joints]
            total = (pose[:, joints] * m[..., None]).sum(axis=1)
            return total / np.maximum(m.sum(axis=1, keepdims=True), 1), m.any(axis=1)

        hip, hip_frames = frame_mean(HIPS)
        shoulder, shoulder_frames = frame_mean(SHOULDERS)
        frames = hip_frames & shoulder_frames
        if not frames.any():
            continue

        torso = np.median(np.linalg.norm(shoulder[frames] - hip[frames], axis=1))
        points = pose[mask]
        diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if diag > 0:
            ratios.append(torso / diag)

    return float(np.median(ratios))
