import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50
def rgb_transform(images):
    if not images:
        raise ValueError("No imaghe")

    transform = transforms.Compose([
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    frames = [
        transform(image.convert("RGB"))
        for image in images
    ]

    return torch.stack(frames, dim=0)