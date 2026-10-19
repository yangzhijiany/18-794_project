# Train / evaluate the pose-only ST-GCN on Penn Action (GT pose)
#
# Same conventions as RGB/train_RGB_classifier.py: one random ratio per video per
# training epoch, all seven ratios on val/test, model selection on the mean val
# accuracy over ratios, outputs under runs/.
#
#   python -m Pose.train_pose train --name bbox_s0 --norm bbox --seed 0
#   python -m Pose.train_pose test  --checkpoint runs/pose/bbox_s0/best.pt --split test

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from data_process.penn_data import PennDataset
from .pose_model import PoseModel
from .pose_transform import calibrate_fallback_k

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/Penn_Action"
SPLITS = ROOT / "splits"


def make_dataset(split):
    return PennDataset(DATA_ROOT, SPLITS / f"{split}.txt", SPLITS / "classes.csv",
                       option="pose", training=(split == "train"), num_frames=16)


def make_loader(split, batch_size, workers, device, dataset=None):
    # Training samples one ratio per video; val/test check all seven ratios
    dataset = dataset or make_dataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=(device.type == "cuda"))


def class_names():
    classes = pd.read_csv(SPLITS / "classes.csv").sort_values("label_id")
    return classes["action"].tolist()


@torch.no_grad()
def validate(model, loader, device, keep_predictions=False):
    model.eval()
    counts = {round(float(r), 1): [0, 0] for r in loader.dataset.ratios}
    total_loss = 0.0
    rows = []

    for batch in loader:
        pose = batch["pose"].to(device)
        visibility = batch["joint_visibility"].to(device)
        target = batch["label"].to(device)

        logits = model(pose, visibility).float()
        loss = cross_entropy(logits, target)
        total_loss += loss.item() * len(target)

        predictions = logits.argmax(dim=1).cpu().tolist()
        labels = batch["label"].tolist()
        ratios = batch["ratio"].tolist()
        for j in range(len(predictions)):
            key = round(ratios[j], 1)
            counts[key][0] += int(predictions[j] == labels[j])
            counts[key][1] += 1

        if keep_predictions:
            probs = logits.softmax(dim=1).cpu()
            for j in range(len(predictions)):
                rows.append((batch["sample_id"][j], round(ratios[j], 1), labels[j],
                             predictions[j], probs[j].tolist()))

    by_ratio = {str(r): c[0] / c[1] for r, c in counts.items()}
    result = {"val_loss": total_loss / len(loader.dataset),
              "mean_accuracy": sum(by_ratio.values()) / len(by_ratio),
              "accuracy_by_ratio": by_ratio}
    if keep_predictions:
        result["rows"] = rows
    return result


def predictions_frame(rows):
    # One row per (video, ratio): sample_id, ratio, label, pred, score_<class> (softmax)
    names = class_names()
    records = []
    for sample_id, ratio, label, pred, probs in rows:
        record = {"sample_id": sample_id, "ratio": ratio, "label": label, "pred": pred}
        record.update({f"score_{name}": p for name, p in zip(names, probs)})
        records.append(record)
    return pd.DataFrame(records)


def fit(model, optimizer, scheduler, epochs, train_loader, val_loader, output_dir,
        device, config, val_every=2, resume=True, max_seconds=None):
    # last.pt is written after every epoch, so an interrupted run continues where it
    # stopped (same RNG state, so the result matches an uninterrupted run)
    output_dir = Path(output_dir)
    last_file = output_dir / "last.pt"
    best_file = output_dir / "best.pt"
    history, best_accuracy, first_epoch = [], -1.0, 1

    if output_dir.exists():
        if not (resume and last_file.exists()):
            raise FileExistsError(f"Experiment already exists: {output_dir}")
        saved_config = json.loads((output_dir / "config.json").read_text())
        if saved_config != json.loads(json.dumps(config)):
            raise ValueError(f"Config differs from the interrupted run in {output_dir}")
        state = torch.load(last_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        history, best_accuracy = state["history"], state["best_accuracy"]
        first_epoch = state["epoch"] + 1
        print(f"Resuming {output_dir.name} from epoch {first_epoch}", flush=True)
    else:
        output_dir.mkdir(parents=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    label_smoothing = config.get("label_smoothing", 0.0)
    start = time.time()

    for epoch in range(first_epoch, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            pose = batch["pose"].to(device)
            visibility = batch["joint_visibility"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(pose, visibility), labels, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_loader.dataset)
        entry = {"epoch": epoch, "train_loss": train_loss}

        if epoch % val_every == 0 or epoch == epochs:
            val = validate(model, val_loader, device)
            if val["mean_accuracy"] > best_accuracy:
                best_accuracy = val["mean_accuracy"]
                torch.save({"model": model.state_dict(), "model_args": model.model_args,
                            "epoch": epoch, "config": config}, best_file)
                (output_dir / "val_metrics.json").write_text(
                    json.dumps({**val, "epoch": epoch}, indent=2))

            entry.update({"val_loss": val["val_loss"], "val_accuracy": val["mean_accuracy"],
                          "best_accuracy": best_accuracy,
                          "accuracy_by_ratio": val["accuracy_by_ratio"]})
            print(f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} | "
                  f"val loss {val['val_loss']:.4f} | val accuracy {val['mean_accuracy']:.2%} | "
                  f"best {best_accuracy:.2%} | {time.time() - start:.0f}s", flush=True)

        history.append(entry)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(),
                    "epoch": epoch, "best_accuracy": best_accuracy, "history": history},
                   last_file.with_suffix(".tmp"))
        last_file.with_suffix(".tmp").replace(last_file)   # atomic: never a half-written last.pt

        # Optional time budget: stop after a finished epoch; running again resumes
        if max_seconds is not None and epoch < epochs and time.time() - start > max_seconds:
            print(f"Stopping after epoch {epoch} (time budget); rerun to resume", flush=True)
            return None

    return best_file


def train_pose(experiment_name, norm_mode="bbox", epochs=120, batch_size=32,
               learning_rate=2e-3, weight_decay=1e-2, base_channels=64,
               label_smoothing=0.1, val_every=2, workers=0, seed=0,
               device_name="cpu", input_mode="xyv", augment=True, max_seconds=None):
    torch.manual_seed(seed)
    device = torch.device(device_name)

    train_set = make_dataset("train")
    fallback_k = calibrate_fallback_k(train_set.samples)   # training split only
    train_loader = make_loader("train", batch_size, workers, device, train_set)
    val_loader = make_loader("val", 256, workers, device)

    model = PoseModel(num_classes=train_set.num_classes, norm_mode=norm_mode,
                      fallback_k=fallback_k, base_channels=base_channels,
                      input_mode=input_mode)
    model.augment = augment
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=epochs * len(train_loader), pct_start=0.1)

    config = {"stage": "pose", "experiment_name": experiment_name, "norm_mode": norm_mode,
              "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
              "weight_decay": weight_decay, "base_channels": base_channels,
              "label_smoothing": label_smoothing, "val_every": val_every,
              "workers": workers, "seed": seed, "input_mode": input_mode,
              "augment": augment, "fallback_k": fallback_k,
              "num_parameters": sum(p.numel() for p in model.parameters())}

    output_dir = ROOT / "runs/pose" / experiment_name
    return fit(model, optimizer, scheduler, epochs, train_loader, val_loader,
               output_dir, device, config, val_every, max_seconds=max_seconds)


def test_pose(checkpoint, split="test", batch_size=256, workers=0, device_name="cpu"):
    # Writes {split}_metrics.json and predictions_{split}.csv next to the checkpoint
    checkpoint = Path(checkpoint)
    device = torch.device(device_name)
    model = PoseModel.from_checkpoint(checkpoint).to(device)

    result = validate(model, make_loader(split, batch_size, workers, device), device,
                      keep_predictions=True)
    frame = predictions_frame(result.pop("rows"))
    frame.to_csv(checkpoint.parent / f"predictions_{split}.csv", index=False)

    result[f"{split}_loss"] = result.pop("val_loss")
    (checkpoint.parent / f"{split}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def export_features(checkpoint, split, batch_size=256, workers=0, device_name="cpu"):
    # Pooled pose features for every (video, ratio) of a split -> features_{split}.pt
    checkpoint = Path(checkpoint)
    device = torch.device(device_name)
    model = PoseModel.from_checkpoint(checkpoint).to(device).eval()

    dataset = make_dataset(split)
    if split == "train":
        dataset.training = False   # export every ratio, not a random one
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    ids, ratios, labels, features = [], [], [], []
    for batch in loader:
        features.append(model.extract_features(batch["pose"].to(device),
                                               batch["joint_visibility"].to(device)).cpu())
        ids += list(batch["sample_id"])
        ratios += batch["ratio"].tolist()
        labels += batch["label"].tolist()

    out = {"sample_id": ids, "ratio": torch.tensor(ratios), "label": torch.tensor(labels),
           "features": torch.cat(features), "checkpoint": str(checkpoint)}
    torch.save(out, checkpoint.parent / f"features_{split}.pt")
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--name", required=True)
    train.add_argument("--norm", default="bbox", choices=["torso", "bbox"])
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--input", default="xyv", choices=["xyv", "xy", "v"])
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--max-seconds", type=float, default=None,
                       help="stop after the epoch that passes this many seconds; run again to resume")

    test = sub.add_parser("test")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--split", default="test", choices=["val", "test"])
    test.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    feats = sub.add_parser("features")
    feats.add_argument("--checkpoint", required=True)
    feats.add_argument("--split", default="test", choices=["train", "val", "test"])
    feats.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "train":
        best = train_pose(args.name, norm_mode=args.norm, epochs=args.epochs,
                          batch_size=args.batch_size, learning_rate=args.lr,
                          weight_decay=args.weight_decay, base_channels=args.width,
                          input_mode=args.input, augment=not args.no_augment,
                          seed=args.seed, workers=args.workers, device_name=args.device,
                          max_seconds=args.max_seconds)
        print("Best checkpoint:", best)
    elif args.command == "test":
        result = test_pose(args.checkpoint, args.split, device_name=args.device)
        print(json.dumps(result, indent=2))
    else:
        out = export_features(args.checkpoint, args.split, device_name=args.device)
        print("features", tuple(out["features"].shape))


if __name__ == "__main__":
    main()
