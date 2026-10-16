from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from .sampling import sample_frame
from .transform import rgb_transform


class PennDataset(Dataset):
    def __init__(
        self,
        data_path,
        split_file,
        classes_file,
        option="fusion",    # choose from rgb, pose or fusion: three model
        training=False,     # True: train, validation, False: Test
        ratios=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
        num_frames=16,
    ):
        self.data_path = Path(data_path)
        self.option = option
        self.training = training
        self.ratios = ratios
        self.num_frames = num_frames

        if option not in ["rgb", "pose", "fusion"]:
            raise ValueError("Invalid option")

        self.video_ids = []
        lines = Path(split_file).read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line in lines:
            video_id = line.strip()
            if video_id:
                self.video_ids.append(video_id)

        classes = pd.read_csv(classes_file)
        self.class_to_idx = dict(
            zip(classes["action"], classes["label_id"])
        )
        self.num_classes = len(classes)
        self.samples = []

        for video_id in self.video_ids:
            label_path = self.data_path / "labels" / f"{video_id}.mat"
            annotation = loadmat(label_path)
            action = str(annotation["action"].item()).strip()

            sample = {
                "video_id": video_id,
                "label": int(self.class_to_idx[action]),
                "total_frames": annotation["x"].shape[0],
            }

            if option in ["rgb", "fusion"]:
                frame_dir = self.data_path / "frames" / video_id

                frame_files = []
                for path in frame_dir.glob("*.jpg"):
                    frame_files.append(path)

                frame_files.sort(key=lambda path: int(path.stem))
                sample["frame_files"] = frame_files

            if option in ["pose", "fusion"]:
                sample["pose"] = np.stack(
                    [annotation["x"], annotation["y"]],
                    axis=-1,
                ).astype(np.float32)

                sample["visibility"] = annotation["visibility"].astype(
                    np.float32
                )

            self.samples.append(sample)

    def __len__(self):
        if self.training:
            return len(self.samples)

        return len(self.samples) * len(self.ratios)

    def __getitem__(self, index):
        if self.training:
            video_index = index
            ratio_index = torch.randint(len(self.ratios), ()).item()    # random choose a ratio in training mode
        else:
            video_index = index // len(self.ratios)
            ratio_index = index % len(self.ratios) # for each video, test each ratio

        sample = self.samples[video_index]
        ratio = self.ratios[ratio_index]

        prefix_length, indices = sample_frame(
            sample["total_frames"],
            ratio,
            self.num_frames,
        )

        output = {
            "sample_id": sample["video_id"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "original_length": sample["total_frames"],
            "prefix_length": prefix_length,
            "frame_indices": torch.from_numpy(indices),
        }

        if self.option in ["rgb", "fusion"]:
            images = []

            for i in indices:
                frame_path = sample["frame_files"][int(i)]

                with Image.open(frame_path) as image:
                    images.append(image.convert("RGB"))

            width, height = images[0].size
            output["original_size"] = torch.tensor([height, width])
            output["rgb"] = rgb_transform(images)

        if self.option in ["pose", "fusion"]:
            output["pose"] = torch.from_numpy(
                sample["pose"][indices]
            )
            output["joint_visibility"] = torch.from_numpy(
                sample["visibility"][indices]
            )

        return output