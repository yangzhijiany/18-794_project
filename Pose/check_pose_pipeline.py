# Checks for the pose pipeline, run from the project root:
#   python -m Pose.check_pose_pipeline
#
# 1 prefix-only    : changing frames after the observed prefix changes nothing
# 2 invariance     : scaling / translating the raw pixels does not change the input
# 3 flip           : flip is an involution and swaps left/right pairs
# 4 reference      : batched torch normalization == simple per-clip numpy version
# 5 integration    : real DataLoader batch -> logits / features of the right shape,
#                    eval() is deterministic and never augments

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .pose_model import PoseModel
from .pose_transform import (
    FLIP_INDEX, HIPS, NORM_MODES, SHOULDERS, augment_pose, flip_pose, normalize_pose,
)
from .train_pose import make_dataset

FALLBACK_K = 0.259


def reference_normalize(pose, visibility, mode, k):
    # One clip at a time, written directly from the definition
    pose = pose.astype(np.float64)
    mask = visibility > 0.5
    out = np.zeros((3,) + mask.shape)
    out[2] = mask
    if not mask.any():
        return out

    points = pose[mask]
    lo, hi = points.min(axis=0), points.max(axis=0)
    box_centre, box_scale = (lo + hi) / 2, k * np.linalg.norm(hi - lo)
    centre, scale = box_centre, box_scale

    if mode == "torso":
        hips, shoulders = [], []
        for t in range(len(pose)):
            hip_vis = [j for j in HIPS if mask[t, j]]
            sh_vis = [j for j in SHOULDERS if mask[t, j]]
            hip = pose[t, hip_vis].mean(axis=0) if hip_vis else None
            sh = pose[t, sh_vis].mean(axis=0) if sh_vis else None
            hips.append(hip)
            if hip is not None and sh is not None:
                shoulders.append(np.linalg.norm(sh - hip))
        hip_points = [h for h in hips if h is not None]
        if hip_points:
            centre = np.mean(hip_points, axis=0)
        if shoulders:
            scale = float(np.median(shoulders))

    if scale <= 1e-3:
        scale = 1.0
    coords = (pose - centre) / scale
    coords[~mask] = 0
    out[0], out[1] = coords[..., 0], coords[..., 1]
    return out


def all_clips(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(loader))


def check_prefix(dataset, model):
    # For each ratio, overwrite every frame after the observed prefix with garbage
    # and compare the clip inputs and logits with the clean ones
    rng = np.random.default_rng(0)
    num_ratios = len(dataset.ratios)
    ok, corrupted_videos, clips = True, 0, 0

    for r, ratio in enumerate(dataset.ratios):
        indices = [v * num_ratios + r for v in range(len(dataset.samples))]
        clean = [dataset[i] for i in indices]

        backup = []
        for sample, item in zip(dataset.samples, clean):
            prefix = int(item["prefix_length"])
            backup.append((sample["pose"].copy(), sample["visibility"].copy()))
            tail = sample["pose"][prefix:]
            sample["pose"][prefix:] = rng.uniform(-500, 1500, tail.shape)
            sample["visibility"][prefix:] = rng.integers(0, 2, sample["visibility"][prefix:].shape)
            corrupted_videos += int(prefix < sample["total_frames"])
        corrupted = [dataset[i] for i in indices]
        for sample, (pose, vis) in zip(dataset.samples, backup):
            sample["pose"], sample["visibility"] = pose, vis

        pose_a = torch.stack([c["pose"] for c in clean])
        vis_a = torch.stack([c["joint_visibility"] for c in clean])
        pose_b = torch.stack([c["pose"] for c in corrupted])
        vis_b = torch.stack([c["joint_visibility"] for c in corrupted])
        with torch.no_grad():
            same_logits = torch.equal(model(pose_a, vis_a), model(pose_b, vis_b))
        ok &= torch.equal(pose_a, pose_b) and torch.equal(vis_a, vis_b) and same_logits
        clips += len(indices)

    # Positive control: the same corruption INSIDE the prefix must change the logits
    item = dataset[len(dataset.ratios) - 1]          # first video, ratio 1.0
    sample = dataset.samples[0]
    pose, vis = sample["pose"].copy(), sample["visibility"].copy()
    sample["pose"] = rng.uniform(-500, 1500, pose.shape).astype(np.float32)
    changed_item = dataset[len(dataset.ratios) - 1]
    sample["pose"], sample["visibility"] = pose, vis
    with torch.no_grad():
        detects = not torch.equal(
            model(item["pose"][None], item["joint_visibility"][None]),
            model(changed_item["pose"][None], changed_item["joint_visibility"][None]))

    return ok and corrupted_videos > 0 and detects, \
        (f"inputs and logits identical={ok} over {clips} clips "
         f"({corrupted_videos} had frames after the prefix); "
         f"positive control (corrupt inside prefix) changes logits={detects}")


def check_invariance(batch):
    pose, vis = batch["pose"], batch["joint_visibility"]
    gen = torch.Generator().manual_seed(1)
    scale = torch.empty(len(pose), 1, 1, 1).uniform_(0.5, 2.0, generator=gen)
    shift = torch.empty(len(pose), 1, 1, 2).uniform_(-300, 300, generator=gen)
    worst = 0.0
    for mode in NORM_MODES:
        a = normalize_pose(pose, vis, mode, FALLBACK_K)
        b = normalize_pose(pose * scale + shift, vis, mode, FALLBACK_K)
        worst = max(worst, (a - b).abs().max().item())
    return worst < 1e-4, f"max |diff| = {worst:.2e}"


def check_flip(batch):
    x = normalize_pose(batch["pose"], batch["joint_visibility"], "torso", FALLBACK_K)
    once, twice = flip_pose(x), flip_pose(flip_pose(x))
    involution = torch.equal(twice, x)
    pairs = all(torch.equal(once[:, 1:, :, FLIP_INDEX[j]], x[:, 1:, :, j]) and
                torch.equal(once[:, 0, :, FLIP_INDEX[j]], -x[:, 0, :, j]) for j in range(13))
    # Augmentation keeps invisible joints at exactly zero
    aug = augment_pose(x)
    invisible = (aug[:, 2:3] == 0).expand_as(aug[:, :2])   # flip permutes joints
    zeros_kept = bool((aug[:, :2][invisible] == 0).all())
    return involution and pairs and zeros_kept, \
        f"involution={involution}, pair swap={pairs}, invisible stay 0={zeros_kept}"


def check_reference(batch):
    pose, vis = batch["pose"], batch["joint_visibility"]
    worst = 0.0
    for mode in NORM_MODES:
        fast = normalize_pose(pose, vis, mode, FALLBACK_K).double().numpy()
        for i in range(len(pose)):
            ref = reference_normalize(pose[i].numpy(), vis[i].numpy(), mode, FALLBACK_K)
            worst = max(worst, float(np.abs(fast[i] - ref).max()))
    return worst < 1e-5, f"max |diff| = {worst:.2e} over {len(pose)} clips x {len(NORM_MODES)} modes"


def check_integration(dataset, model):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    model.augment = True
    model.eval()
    with torch.no_grad():
        logits = model(batch["pose"], batch["joint_visibility"])
        again = model(batch["pose"], batch["joint_visibility"])
        features = model.extract_features(batch["pose"], batch["joint_visibility"])
    model.augment = False
    ok = (tuple(logits.shape) == (8, dataset.num_classes)
          and tuple(features.shape) == (8, model.feature_dim)
          and torch.equal(logits, again))
    return ok, (f"pose {tuple(batch['pose'].shape)}, visibility "
                f"{tuple(batch['joint_visibility'].shape)} -> logits {tuple(logits.shape)}, "
                f"features {tuple(features.shape)}, deterministic={torch.equal(logits, again)}")


def main():
    torch.manual_seed(0)
    dataset = make_dataset("val")
    model = PoseModel(num_classes=dataset.num_classes, fallback_k=FALLBACK_K).eval()
    batch = all_clips(dataset)

    checks = [
        ("1 prefix-only", lambda: check_prefix(dataset, model)),
        ("2 invariance", lambda: check_invariance(batch)),
        ("3 flip", lambda: check_flip(batch)),
        ("4 reference", lambda: check_reference(batch)),
        ("5 integration", lambda: check_integration(dataset, model)),
    ]
    failed = False
    for name, run in checks:
        ok, detail = run()
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
