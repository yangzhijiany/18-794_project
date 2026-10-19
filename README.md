# 18-794_project



## RGB

### Trained RGB model weights

Our classifier and fine-tuned model checkpoints are available in [rgb_results.tar](https://cmu.box.com/s/5ekkve7t6nt1pkzr5fufjczmavwbuq9v).

### Pretrained ResNet-50 + GSF weights

Download the [16-frame ResNet-50 + GSF checkpoint](https://drive.google.com/file/d/1nSjhYvsu56e7a9d9VasGo7bxOrJ5j05f/view?usp=sharing) provided by the [GSF authors](https://github.com/swathikirans/GSF). Save it as `something-v1_resnet50_16frames.pth.tar` in the project root to train the RGB model.

## Pose

Pose-only model: ST-GCN on the 13 ground-truth Penn Action joints. Code is in `Pose/`.

### Joints and visibility

Joint order in the label files: 0 head, then (right, left) pairs: 1/2 shoulder, 3/4 elbow, 5/6 wrist, 7/8 hip, 9/10 knee, 11/12 ankle. Odd indices are the person's right side, so when mapping COCO keypoints to Penn, COCO `right_shoulder` goes to index 1, `left_shoulder` to index 2, and so on.

The coordinates of joints with `visibility = 0` should not be used (many are placeholders near (0, 0) or copies of the previous frame). We only use visible joints for the statistics, set invisible joints to 0 after normalization and give the visibility to the model as a third channel.

### Preprocessing

Normalization is done inside `PoseModel.forward()`, so the model takes the Dataset output directly:

```python
from Pose.pose_model import PoseModel
model = PoseModel.from_checkpoint("runs/pose/<run>/best.pt").eval()
logits = model(batch["pose"], batch["joint_visibility"])                     # [B, 15]
features = model.extract_features(batch["pose"], batch["joint_visibility"])  # [B, 256]
```

For each clip we only use the 16 sampled frames (they are all inside the observed prefix) and only the visible joints.

- `bbox` (default): centre and scale from the box around the visible joints. The scale is the box diagonal times `fallback_k` (0.259, computed on the training split so that one unit is about one torso length).
- `torso`: centre at the hips, scale = median shoulder-to-hip distance, falls back to the box when the hips or torso are not visible.

The input is `[B, 3, 16, 13]` (x, y, visibility). Training uses horizontal flip (with left/right swap), rotation of up to 15 degrees and scaling between 0.9 and 1.1. `extract_features` returns the pooled 256-d feature before dropout and the classifier (not L2-normalized).

### Training and evaluation

```bash
python -m Pose.check_pose_pipeline
python -m Pose.train_pose train --name bbox_s0 --norm bbox --seed 0
python -m Pose.train_pose test --checkpoint runs/pose/bbox_s0/best.pt --split test
python -m Pose.train_pose features --checkpoint runs/pose/bbox_s0/best.pt --split train
```

Pose training only needs `data/Penn_Action/labels/` and runs on CPU (about one hour per run on one core). If a run is interrupted, running the same command again resumes from `last.pt`. The best checkpoint is chosen by the mean val accuracy over the seven ratios, same as the RGB model.

`test` saves `test_metrics.json` and `predictions_test.csv` next to the checkpoint, with one row per (video, ratio): `sample_id`, `ratio`, `label`, `pred` and `score_<class>` (softmax). Read `sample_id` as a string to keep the leading zeros.

### Results

Test accuracy (%) with ground-truth joints, mean and standard deviation over three seeds:

| Ratio | 10% | 20% | 30% | 40% | 60% | 80% | 100% | Mean |
|---|---|---|---|---|---|---|---|---|
| Pose (ST-GCN) | 88.4 (0.6) | 91.1 (0.1) | 93.1 (0.2) | 94.0 (0.3) | 96.3 (0.3) | 97.7 (0.2) | 97.4 (0.1) | 94.0 (0.2) |

The checkpoint used for fusion is seed 2 (`pose_bbox_s2_best.pt`, val mean 94.86%, feature dimension 256). It is shared together with its prediction CSVs and the pooled features for every (video, ratio) of train/val/test (`features_{split}.pt` with `sample_id`, `ratio`, `label`, `features`).
