"""Multimodal fusion models for SmashClip."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models import MLPHead, SinusoidalPE


class _ModalityEncoder(nn.Module):
    """Encode each modality to a common proj_dim, with LayerNorm."""

    def __init__(self, task: str, proj_dim: int = 256, visual_dim: int = 768) -> None:
        super().__init__()
        self.task = task.upper()
        self.proj_dim = proj_dim

        # V1: Transformer path
        self.v1_pe = SinusoidalPE(visual_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim, nhead=8, dim_feedforward=visual_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.v1_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.v1_proj = nn.Sequential(nn.Linear(visual_dim, proj_dim), nn.ReLU())
        self.v1_ln = nn.LayerNorm(proj_dim)

        # A1: BEATs audio
        self.a1_proj = nn.Sequential(nn.Linear(768, proj_dim), nn.ReLU())
        self.a1_ln = nn.LayerNorm(proj_dim)

        # A2: Transcript embedding
        self.a2_proj = nn.Sequential(nn.Linear(768, proj_dim), nn.ReLU())
        self.a2_ln = nn.LayerNorm(proj_dim)

        # A3: Prosody
        self.a3_bn = nn.BatchNorm1d(12)
        self.a3_proj = nn.Sequential(nn.Linear(12, proj_dim), nn.ReLU())
        self.a3_ln = nn.LayerNorm(proj_dim)

        # M1: Metadata embeddings
        self.m1_killer_emb = nn.Embedding(86, 32)
        self.m1_victim_emb = nn.Embedding(86, 32)
        self.m1_move_emb = nn.Embedding(24, 32)
        self.m1_stage_emb = nn.Embedding(8, 32)
        m1_in = 128 + (25 if self.task == "A" else 0)
        self.m1_proj = nn.Sequential(nn.Linear(m1_in, proj_dim), nn.ReLU())
        self.m1_ln = nn.LayerNorm(proj_dim)

    def _encode_v1_seq(self, batch: dict) -> torch.Tensor:
        """Return V1 Transformer output sequence (B, T, proj_dim)."""
        x = batch["internvideo2"]
        mask = batch["internvideo2_mask"]
        x = self.v1_pe(x)
        x = self.v1_transformer(x, src_key_padding_mask=~mask)
        x = self.v1_proj(x)
        x = self.v1_ln(x)
        return x

    def _pool_v1(self, v1_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pool over V1 sequence."""
        mask_f = mask.unsqueeze(-1).float()
        return (v1_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def encode(self, batch: dict, modality: str) -> torch.Tensor:
        """Encode a single modality to (B, proj_dim)."""
        if modality == "V1":
            seq = self._encode_v1_seq(batch)
            return self._pool_v1(seq, batch["internvideo2_mask"])
        elif modality == "A1":
            return self.a1_ln(self.a1_proj(batch["beats_audio"]))
        elif modality == "A2":
            return self.a2_ln(self.a2_proj(batch["text_embed"]))
        elif modality == "A3":
            return self.a3_ln(self.a3_proj(self.a3_bn(batch["prosody"])))
        elif modality == "M1":
            k = self.m1_killer_emb(batch["killer_id"])
            v = self.m1_victim_emb(batch["victim_id"])
            m = self.m1_move_emb(batch["move_id"])
            s = self.m1_stage_emb(batch["stage_id"])
            parts = [k, v, m, s]
            if self.task == "A":
                parts.append(batch["scene_tags"])
            x = torch.cat(parts, dim=-1)
            return self.m1_ln(self.m1_proj(x))
        else:
            raise ValueError(f"Unknown modality: {modality}")


# ── F1: Late Fusion (concat) ──


class LateFusionModel(nn.Module):
    """Late fusion: project each modality to proj_dim, concat, MLP head."""

    def __init__(
        self,
        modalities: list[str],
        task: str,
        output_dim: int,
        proj_dim: int = 256,
        visual_dim: int = 768,
    ) -> None:
        super().__init__()
        self.modalities = sorted(modalities)
        self.encoder = _ModalityEncoder(task, proj_dim, visual_dim)
        concat_dim = proj_dim * len(self.modalities)
        self.head = MLPHead(concat_dim, output_dim, hidden=512, dropout=0.3)

    def forward(self, batch: dict) -> torch.Tensor:
        features = [self.encoder.encode(batch, m) for m in self.modalities]
        x = torch.cat(features, dim=-1)
        return self.head(x)


# ── F3: Cross-Attention Fusion ──


class CrossAttentionFusionModel(nn.Module):
    """Cross-attention: V1 sequence queries other modalities.

    Query = V1 Transformer output sequence (B, T, proj_dim)
    Key/Value = [A1, A2, A3, M1] projected to (B, 4, proj_dim)
    2-layer cross-attention with 8 heads.
    """

    def __init__(
        self,
        task: str,
        output_dim: int,
        proj_dim: int = 256,
        visual_dim: int = 768,
    ) -> None:
        super().__init__()
        self.encoder = _ModalityEncoder(task, proj_dim, visual_dim)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=proj_dim, num_heads=8, batch_first=True)
            for _ in range(2)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(proj_dim) for _ in range(2)
        ])
        self.head = MLPHead(proj_dim, output_dim, hidden=512, dropout=0.3)

    def forward(self, batch: dict) -> torch.Tensor:
        query = self.encoder._encode_v1_seq(batch)
        mask = batch["internvideo2_mask"]

        kv_list = [
            self.encoder.encode(batch, m) for m in ["A1", "A2", "A3", "M1"]
        ]
        kv = torch.stack(kv_list, dim=1)

        x = query
        for attn, norm in zip(self.cross_attn_layers, self.cross_norms):
            attn_out, _ = attn(query=x, key=kv, value=kv, key_padding_mask=None)
            x = norm(x + attn_out)

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


# ── F2: Early Fusion (shared Transformer with modality tags) ──


class EarlyFusionModel(nn.Module):
    """Shared Transformer encoder with modality tag tokens.

    Each modality is projected to proj_dim, tagged with a learnable modality
    embedding, and all tokens are fed into a shared Transformer encoder.
    A CLS token is prepended and its output is used for prediction.

    V1 (visual): each frame becomes a token.
    A1/A2/A3/M1: each becomes a single token.
    """

    _MODALITY_TAG_IDS = {"V1": 0, "A1": 1, "A2": 2, "A3": 3, "M1": 4, "CLS": 5}

    def __init__(
        self,
        modalities: list[str],
        task: str,
        output_dim: int,
        visual_dim: int = 768,
        proj_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.modalities = sorted(modalities)
        self.task = task.upper()
        self.proj_dim = proj_dim

        self.projections = nn.ModuleDict()
        if "V1" in self.modalities:
            self.projections["V1"] = nn.Sequential(
                nn.Linear(visual_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
            )
        if "A1" in self.modalities:
            self.projections["A1"] = nn.Sequential(
                nn.Linear(768, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
            )
        if "A2" in self.modalities:
            self.projections["A2"] = nn.Sequential(
                nn.Linear(768, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
            )
        if "A3" in self.modalities:
            self.a3_bn = nn.BatchNorm1d(12)
            self.projections["A3"] = nn.Sequential(
                nn.Linear(12, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
            )
        if "M1" in self.modalities:
            self.m1_killer_emb = nn.Embedding(86, 32)
            self.m1_victim_emb = nn.Embedding(86, 32)
            self.m1_move_emb = nn.Embedding(24, 32)
            self.m1_stage_emb = nn.Embedding(8, 32)
            m1_in = 128 + (25 if self.task == "A" else 0)
            self.projections["M1"] = nn.Sequential(
                nn.Linear(m1_in, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
            )

        self.modality_tags = nn.Embedding(6, proj_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = MLPHead(proj_dim, output_dim, hidden=head_hidden, dropout=head_dropout)

    def _get_m1_input(self, batch: dict) -> torch.Tensor:
        k = self.m1_killer_emb(batch["killer_id"])
        v = self.m1_victim_emb(batch["victim_id"])
        m = self.m1_move_emb(batch["move_id"])
        s = self.m1_stage_emb(batch["stage_id"])
        parts = [k, v, m, s]
        if self.task == "A":
            parts.append(batch["scene_tags"])
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        B = batch["internvideo2"].size(0) if "V1" in self.modalities else batch["beats_audio"].size(0)
        device = next(self.parameters()).device

        tokens = []
        mask_parts = []

        for mod in self.modalities:
            tag_id = self._MODALITY_TAG_IDS[mod]
            tag = self.modality_tags(torch.tensor(tag_id, device=device))

            if mod == "V1":
                x = self.projections["V1"](batch["internvideo2"])
                x = x + tag.unsqueeze(0).unsqueeze(0)
                tokens.append(x)
                mask_parts.append(batch["internvideo2_mask"])
            elif mod == "A3":
                x = self.a3_bn(batch["prosody"])
                x = self.projections["A3"](x).unsqueeze(1)
                x = x + tag.unsqueeze(0).unsqueeze(0)
                tokens.append(x)
                mask_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))
            elif mod == "M1":
                raw = self._get_m1_input(batch)
                x = self.projections["M1"](raw).unsqueeze(1)
                x = x + tag.unsqueeze(0).unsqueeze(0)
                tokens.append(x)
                mask_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))
            else:
                key = "beats_audio" if mod == "A1" else "text_embed"
                x = self.projections[mod](batch[key]).unsqueeze(1)
                x = x + tag.unsqueeze(0).unsqueeze(0)
                tokens.append(x)
                mask_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        cls_tag_id = self._MODALITY_TAG_IDS["CLS"]
        cls_tag = self.modality_tags(torch.tensor(cls_tag_id, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_tag.unsqueeze(0).unsqueeze(0)
        tokens.insert(0, cls)
        mask_parts.insert(0, torch.ones(B, 1, dtype=torch.bool, device=device))

        x = torch.cat(tokens, dim=1)
        mask = torch.cat(mask_parts, dim=1)
        x = self.transformer(x, src_key_padding_mask=~mask)
        return self.head(x[:, 0])


# ── Factory ──


def build_fusion_model(
    name: str,
    modalities: list[str],
    task: str,
    output_dim: int,
    visual_dim: int = 768,
) -> nn.Module:
    """Build a fusion model by name.

    Parameters
    ----------
    name : F1, F2, or F3
    modalities : list of modality keys (used by F1 and F2)
    task : "A" or "B"
    output_dim : 1 for regression, 5 for classification, 20 for scene tags
    visual_dim : visual feature dimension (default: 768)
    """
    if name == "F1":
        return LateFusionModel(modalities, task, output_dim, visual_dim=visual_dim)
    elif name == "F2":
        return EarlyFusionModel(modalities, task, output_dim, visual_dim=visual_dim)
    elif name == "F3":
        return CrossAttentionFusionModel(task, output_dim, visual_dim=visual_dim)
    else:
        raise ValueError(f"Unknown fusion model: {name}")
