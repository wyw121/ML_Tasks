#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model definitions:
- VideoTransformerClassifier: ResNet18 frame encoder + Transformer encoder for temporal modeling + linear head.
- Sinusoidal positional encoding.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class VideoTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_frames: int = 8,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_cnn: bool = False,
    ) -> None:
        super().__init__()
        # ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        modules = list(backbone.children())[:-1]  # remove fc
        self.cnn = nn.Sequential(*modules)  # outputs (B, 512, 1, 1)
        self.cnn_out_dim = backbone.fc.in_features  # 512
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.pos_encoder = SinusoidalPositionalEncoding(self.cnn_out_dim, max_len=num_frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(self.cnn_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)  # (B*T, 512, 1, 1)
        feats = feats.view(b, t, self.cnn_out_dim)
        feats = self.pos_encoder(feats)
        feats = self.transformer(feats)
        feats = feats.mean(dim=1)  # temporal mean pooling
        logits = self.cls_head(feats)
        return logits


def build_model(num_classes: int = 2, num_frames: int = 8, freeze_cnn: bool = False) -> nn.Module:
    return VideoTransformerClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        freeze_cnn=freeze_cnn,
    )
