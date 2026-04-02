"""Unimodal baseline models for SmashClip."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Shared components ──


class MLPHead(nn.Module):
    """Two-layer MLP classifier."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden: int = 256, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


# ── V1: Visual Temporal (Transformer + mean pool) ──


class VisualTemporalModel(nn.Module):
    """1-layer Transformer encoder with mean pooling over visual features."""

    def __init__(self, output_dim: int, visual_dim: int = 768) -> None:
        super().__init__()
        d_model = visual_dim
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = MLPHead(d_model, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["internvideo2"]          # (B, T, D)
        mask = batch["internvideo2_mask"]   # (B, T) True=valid

        x = self.pe(x)
        x = self.transformer(x, src_key_padding_mask=~mask)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


# ── V2: Visual Mean Pool ──


class VisualMeanPoolModel(nn.Module):
    """Masked mean pooling over visual features (no Transformer)."""

    def __init__(self, output_dim: int, visual_dim: int = 768) -> None:
        super().__init__()
        self.head = MLPHead(visual_dim, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["internvideo2"]
        mask = batch["internvideo2_mask"]
        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


# ── A1: Audio (BEATs) ──


class AudioBeatsModel(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.head = MLPHead(768, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(batch["beats_audio"])


# ── A2: Transcript Embedding ──


class AudioTextModel(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.head = MLPHead(768, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(batch["text_embed"])


# ── A3: Prosody ──


class ProsodyModel(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(12)
        self.head = MLPHead(12, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.bn(batch["prosody"])
        return self.head(x)


# ── M1: Metadata ──


class MetadataModel(nn.Module):
    """Categorical embeddings for killer/victim/move/stage.

    Task A: includes scene_tags (20d) as additional input.
    Task C: excludes scene_tags to prevent label leakage.
    """

    def __init__(self, task: str, output_dim: int) -> None:
        super().__init__()
        self.task = task.upper()
        self.killer_emb = nn.Embedding(86, 32)
        self.victim_emb = nn.Embedding(86, 32)
        self.move_emb = nn.Embedding(24, 32)
        self.stage_emb = nn.Embedding(8, 32)

        emb_dim = 128  # 32 * 4
        if self.task == "A":
            emb_dim += 20  # scene_tags
        self.head = MLPHead(emb_dim, output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        k = self.killer_emb(batch["killer_id"])
        v = self.victim_emb(batch["victim_id"])
        m = self.move_emb(batch["move_id"])
        s = self.stage_emb(batch["stage_id"])
        parts = [k, v, m, s]

        if self.task == "A":
            parts.append(batch["scene_tags"])

        x = torch.cat(parts, dim=-1)
        return self.head(x)


# ── MetadataPredictionModel (Task B) ──


class MetadataPredictionModel(nn.Module):
    """Predict metadata (killer/victim/stage/move) from visual features.

    Uses a 1-layer Transformer encoder with a CLS token.
    """

    def __init__(self, num_classes: int, visual_dim: int = 768) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=visual_dim, nhead=8,
                dim_feedforward=visual_dim * 4, dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, visual_dim))
        self.head = MLPHead(visual_dim, num_classes)

    def forward(self, visual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = visual.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, visual], dim=1)
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            x = self.transformer(x, src_key_padding_mask=~full_mask)
        else:
            x = self.transformer(x)
        return self.head(x[:, 0])


# ── Factory ──

_MODEL_REGISTRY: dict[str, type] = {
    "V1": VisualTemporalModel,
    "V2": VisualMeanPoolModel,
    "A1": AudioBeatsModel,
    "A2": AudioTextModel,
    "A3": ProsodyModel,
    "M1": MetadataModel,
}


def build_model(
    name: str, task: str, output_dim: int, visual_dim: int = 768,
) -> nn.Module:
    """Build a model by name.

    Parameters
    ----------
    name : str
        Model identifier (V1, V2, A1, A2, A3, M1).
    task : str
        "A" or "B".
    output_dim : int
        Number of output units.
    visual_dim : int
        Dimension of visual features (default: 768).
    """
    cls = _MODEL_REGISTRY[name]
    if name == "M1":
        return cls(task=task, output_dim=output_dim)
    if name in ("V1", "V2"):
        return cls(output_dim=output_dim, visual_dim=visual_dim)
    return cls(output_dim=output_dim)
