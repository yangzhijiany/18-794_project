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

### Trained pose model weights

Our checkpoints, prediction CSVs and pooled features are available in [pose_results.tar](https://cmu.box.com/s/come21aqa8wrw0nr2seolgb4g9sx2frc). Extract it in the project root to get `runs/pose/bbox_s0`, `bbox_s1` and `bbox_s2`. The checkpoint used for fusion is `runs/pose/bbox_s2/best.pt` (val mean 94.86%, feature dimension 256). The same folder has `features_{train,val,test}.pt` with the pooled features for every (video, ratio) (`sample_id`, `ratio`, `label`, `features`).

## Fusion

We L2-normalize and concatenate frozen RGB (2048-d) and pose (256-d) features, then train FC(2304, 256), ReLU and a 15-class classifier. Training uses AdamW for 100 epochs (batch size 64, lr 0.001, weight decay 0.0001), selecting the checkpoint by mean validation accuracy. Code is in `Fusion/`.

```bash
python -m Fusion.train_fusion train --rgb-dir rgb_results/runs --output runs/fusion/matched_s0 --seed 0
python -m Fusion.train_fusion test --rgb-dir rgb_results/runs --checkpoint runs/fusion/matched_s0/best.pt
```

The observation ratio is the fraction of the video prefix available to both encoders. We evaluate the same fusion head at each ratio. For the ablation, use `--mode shuffled` with a new output directory to pair pose with another video at the same ratio.

Test accuracy (%) at seven observation ratios (GT pose; fusion seed 0, pose baseline `bbox_s2`):

| Method | 10% | 20% | 30% | 40% | 60% | 80% | 100% | Mean | nAUC |
|---|---|---|---|---|---|---|---|---|---|
| RGB | 81.84 | 88.30 | 90.73 | 91.95 | 93.45 | 94.29 | 94.94 | 90.78 | 0.9203 |
| Pose | 89.04 | 91.10 | 92.88 | 94.38 | 96.63 | 97.75 | 97.47 | 94.18 | 0.9515 |
| Feature fusion | 93.73 | 95.13 | 96.72 | 96.63 | 98.13 | 98.69 | 98.69 | 96.82 | 0.9733 |
| Shuffled pose | 80.62 | 87.64 | 90.17 | 90.82 | 92.79 | 93.16 | 93.73 | 89.85 | 0.9111 |

nAUC is the trapezoidal area over 10%-100%, divided by 0.9, with accuracy on a 0-1 scale. The original five-ratio proposal results are also saved in `auc.csv`. Run `python -m Fusion.analyze_fusion` to generate the curve, 10%/20% confusion matrices, class analysis and observed-prefix frame examples in `runs/fusion_analysis_rerun/`.

### Trained fusion model weights

Download [fusion_results.tar](https://cmu.box.com/s/t31mw8ybixp1jqjzpdvz89n6v6egxy5e) for checkpoints, configs, predictions and analysis CSVs. Extract it in the project root; the feature fusion checkpoint is `runs/fusion/matched_s0_rerun/best.pt`.
