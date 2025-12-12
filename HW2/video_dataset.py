#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video dataset utilities for binary classification and captioning.
- Loads video paths and labels from CSV.
- Samples a fixed number of frames uniformly using OpenCV.
- Applies torchvision transforms for ResNet18 input.
"""

import os
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _default_transform(image_size: int) -> Callable:
    """Standard ImageNet normalization pipeline for numpy RGB frames."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _read_frames_opencv(video_path: str, num_frames: int) -> List[np.ndarray]:
    """Uniformly sample frames with OpenCV.

    Args:
        video_path: path to the video file.
        num_frames: number of frames to sample.
    Returns:
        List of BGR frames (numpy arrays). Falls back to fewer frames if short.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has zero frames: {video_path}")

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            # attempt to read next available frame
            ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    cap.release()
    return frames


class VideoDataset(Dataset):
    """Video dataset for binary classification or test-time inference."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        num_frames: int = 8,
        image_size: int = 224,
        has_label: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        import pandas as pd

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.has_label = has_label
        self.transform = transform or _default_transform(image_size)

        if has_label and "label" not in self.df.columns:
            raise ValueError("CSV must contain a 'label' column for training.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        rel_path = row.iloc[0]
        label = row.iloc[1] if self.has_label else -1
        video_path = os.path.join(self.root_dir, rel_path)

        frames = _read_frames_opencv(video_path, self.num_frames)
        if len(frames) == 0:
            raise RuntimeError(f"No frames read from {video_path}")
        # If video shorter than requested, repeat last frame to fixed length
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        processed = []
        for frame in frames:
            # OpenCV uses BGR; convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed.append(self.transform(frame_rgb))
        # shape: (T, 3, H, W)
        clip = torch.stack(processed, dim=0)

        if self.has_label:
            return clip, torch.tensor(label, dtype=torch.long), video_path
        return clip, torch.tensor(-1, dtype=torch.long), video_path


def split_train_val(csv_file: str, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    import pandas as pd

    df = pd.read_csv(csv_file)
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    return indices[:split], indices[split:]
