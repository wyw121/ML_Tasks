#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video binary classification with ResNet18 + Transformer encoder.
Optional: BLIP captions for sampled frames.

Usage examples:
    # Train with defaults (epochs=5) and evaluate
    python train.py --data-dir Data --epochs 5

    # Inference on test set using a checkpoint
    python train.py --data-dir Data --eval --checkpoint outputs/best_seq.pt

    # Generate BLIP captions for test videos (no training required)
    python train.py --data-dir Data --generate-captions --caption-output captions.jsonl --caption-samples 4
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from video_dataset import VideoDataset, split_train_val
from modeling import build_model


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(args) -> Tuple[DataLoader, DataLoader]:
    csv_train = os.path.join(args.data_dir, "labels_train.csv")
    train_idx, val_idx = split_train_val(csv_train, val_ratio=args.val_ratio, seed=args.seed)

    full_dataset = VideoDataset(
        csv_file=csv_train,
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        has_label=True,
    )
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_test_loader(args) -> DataLoader:
    csv_test = os.path.join(args.data_dir, "labels_test.csv")
    test_ds = VideoDataset(
        csv_file=csv_test,
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        has_label=False,
    )
    return DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for clips, labels, _ in tqdm(loader, desc="train", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for clips, labels, _ in tqdm(loader, desc="val", leave=False):
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            total_loss += loss.item() * clips.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


def inference(model, loader, device, output_csv: str):
    import pandas as pd

    model.eval()
    records: List[Dict] = []
    with torch.no_grad():
        for clips, _, paths in tqdm(loader, desc="inference", leave=False):
            clips = clips.to(device)
            logits = model(clips)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            for path, pred in zip(paths, preds):
                rel_path = os.path.relpath(path, start=os.path.dirname(loader.dataset.root_dir))
                records.append({"video_path": rel_path.replace('..\\', '').replace('../', ''), "label": int(pred)})
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def maybe_load_checkpoint(model, checkpoint: str, device):
    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print(f"Loaded checkpoint from {checkpoint}")


def generate_captions(args):
    """Generate captions for sampled frames via BLIP."""
    from transformers import pipeline
    from video_dataset import _read_frames_opencv

    device = 0 if torch.cuda.is_available() else -1
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)

    csv_test = os.path.join(args.data_dir, "labels_test.csv")
    import pandas as pd

    df = pd.read_csv(csv_test)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="caption"):
        rel_path = row[0]
        video_path = os.path.join(args.data_dir, rel_path)
        frames = _read_frames_opencv(video_path, num_frames=args.caption_samples)
        captions = []
        for frame in frames:
            # frame BGR -> RGB
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = captioner(rgb)[0]["generated_text"]
            captions.append(text)
        results.append({"video_path": rel_path, "captions": captions})

    out_path = args.caption_output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Captions saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Data", help="Root data dir containing labels_*.csv and videos_* folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to load checkpoint for eval/inference")
    parser.add_argument("--output-csv", type=str, default="outputs/pred_test.csv")
    parser.add_argument("--eval", action="store_true", help="Eval/inference only (no training)")
    parser.add_argument("--freeze-cnn", action="store_true", help="Freeze ResNet backbone")
    parser.add_argument("--generate-captions", action="store_true", help="Run BLIP captioning on test set")
    parser.add_argument("--caption-output", type=str, default="outputs/captions.jsonl")
    parser.add_argument("--caption-samples", type=int, default=4, help="Frames to caption per video")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.generate_captions:
        generate_captions(args)
        return

    model = build_model(num_classes=2, num_frames=args.num_frames, freeze_cnn=args.freeze_cnn).to(device)
    maybe_load_checkpoint(model, args.checkpoint, device)

    if args.eval and args.checkpoint:
        # Inference only
        test_loader = get_test_loader(args)
        inference(model, test_loader, device, args.output_csv)
        return

    # Training flow
    train_loader, val_loader = get_loaders(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    os.makedirs("outputs", exist_ok=True)
    best_ckpt = Path("outputs/best_seq.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"Saved best checkpoint to {best_ckpt} (val_acc={best_acc:.4f})")

    # Load best and run inference on test
    if best_ckpt.exists():
        maybe_load_checkpoint(model, str(best_ckpt), device)
    test_loader = get_test_loader(args)
    inference(model, test_loader, device, args.output_csv)


if __name__ == "__main__":
    main()
