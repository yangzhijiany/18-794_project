# Train classifier with frozen backbone
import json
from pathlib import Path
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from data_process.penn_data import PennDataset
from .RGB_model import RGBModel, load_pretrained


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/Penn_Action"
PRETRAINED = ROOT / "something-v1_resnet50_16frames.pth.tar"


def make_loader(split, batch_size, workers, device):
    # Training samples one ratio per video
    # Validation checks all seven ratios
    dataset = PennDataset(DATA_ROOT, ROOT / "splits" / f"{split}.txt", ROOT / "splits/classes.csv",
                          option="rgb", training=(split == "train"), num_frames=16)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=(device.type == "cuda"))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    counts = {}

    for i in loader.dataset.ratios:
        counts[round(float(i), 1)] = [0, 0]

    total_loss = 0.0

    for i, batch in enumerate(loader, 1):
        rgb = batch["rgb"].to(device)
        target = batch["label"].to(device)

        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(rgb)
            loss = cross_entropy(logits, target)

        total_loss += loss.item() * len(target)
        predictions = logits.argmax(dim=1).cpu().tolist()
        labels = batch["label"].tolist()
        ratios = batch["ratio"].tolist()

        for j in range(len(predictions)):
            key = round(ratios[j], 1)
            counts[key][0] += int(predictions[j] == labels[j])
            counts[key][1] += 1

    by_ratio = {}
    for ratio, values in counts.items():
        by_ratio[str(ratio)] = values[0] / values[1]

    mean_accuracy = sum(by_ratio.values()) / len(by_ratio)
    return {"val_loss": total_loss / len(loader.dataset),
            "mean_accuracy": mean_accuracy, "accuracy_by_ratio": by_ratio}


def fit(model, optimizer, epochs, batch_size, workers, output_dir, device, config):
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError(f"Experiment already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_loader = make_loader("train", batch_size, workers, device)
    val_loader = make_loader("val", batch_size, workers, device)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = []
    best_accuracy = -1.0
    best_file = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(train_loader, 1):
            rgb = batch["rgb"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(rgb)
                loss = cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_loader.dataset)
        val = validate(model, val_loader, device)

        if val["mean_accuracy"] > best_accuracy:
            best_accuracy = val["mean_accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_file)
            (output_dir / "val_metrics.json").write_text(json.dumps(val, indent=2))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["val_loss"],
                        "val_accuracy": val["mean_accuracy"], "best_accuracy": best_accuracy,
                        "accuracy_by_ratio": val["accuracy_by_ratio"]})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(
            f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} | "
            f"val loss {val['val_loss']:.4f} | "
            f"val accuracy {val['mean_accuracy']:.2%} | best {best_accuracy:.2%}",
            flush=True,
        )

    return best_file


def train_classifier(experiment_name, epochs, batch_size, learning_rate,
                     weight_decay=1e-4, workers=2, seed=666, device_name="cuda"):
    torch.manual_seed(seed)
    device = torch.device(device_name)

    model = RGBModel()
    load_pretrained(model, PRETRAINED)
    model.freeze_backbone(True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.new_fc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    config = {"stage": "classifier", "experiment_name": experiment_name, "epochs": epochs,
              "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay,
              "workers": workers, "seed": seed}

    output_dir = ROOT / "runs/rgb_classifier" / experiment_name
    return fit(model, optimizer, epochs, batch_size, workers, output_dir, device, config)