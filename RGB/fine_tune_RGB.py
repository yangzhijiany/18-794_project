# Fine-tune the GSF backbone and Penn Action classifier together
from pathlib import Path
import torch
from .RGB_model import RGBModel
from .train_RGB_classifier import ROOT, fit


def fine_tune_rgb(experiment_name, classifier_checkpoint, epochs, batch_size,
                  backbone_lr, classifier_lr, weight_decay=1e-4, workers=2,
                  seed=666, device_name="cuda"):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    classifier_checkpoint = Path(classifier_checkpoint)

    checkpoint = torch.load(classifier_checkpoint, map_location="cpu", weights_only=True)

    model = RGBModel()
    model.load_state_dict(checkpoint["model"])
    model.freeze_backbone(False)
    model.to(device)

    # Update both parts, using a separate learning rate for each
    optimizer = torch.optim.AdamW(
        [{"params": model.base_model.parameters(), "lr": backbone_lr},
         {"params": model.new_fc.parameters(), "lr": classifier_lr}],
        weight_decay=weight_decay,
    )

    config = {"stage": "fine_tune", "experiment_name": experiment_name,
              "classifier_checkpoint": str(classifier_checkpoint), "epochs": epochs,
              "batch_size": batch_size, "backbone_lr": backbone_lr,
              "classifier_lr": classifier_lr, "weight_decay": weight_decay,
              "workers": workers, "seed": seed}

    output_dir = ROOT / "runs/rgb_finetune" / experiment_name
    return fit(model, optimizer, epochs, batch_size, workers, output_dir, device, config)