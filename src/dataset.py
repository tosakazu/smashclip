"""Dataset and collate function for SmashClip baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_annotations(path: Path) -> dict[str, dict]:
    """Load JSONL annotation file into {id: record} dict."""
    records: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


# ── Prosody feature extraction ──

_PROSODY_KEYS = [
    ("pitch", "mean"),
    ("pitch", "max"),
    ("pitch", "std"),
    ("pitch", "max_mean_ratio"),
    ("energy", "mean"),
    ("energy", "max"),
    ("energy", "std"),
    ("energy", "rate_of_change"),
    ("speech_rate", "chars_per_sec"),
    ("speech_rate", "words_per_sec"),
    ("speech_rate", "segment_count"),
    ("spectral_centroid", "mean"),
]


def _load_prosody(path: Path) -> np.ndarray:
    """Load prosody JSON and extract 12-dim scalar vector."""
    with open(path) as f:
        d = json.load(f)
    return np.array([d[group][key] for group, key in _PROSODY_KEYS], dtype=np.float32)


# ── Dataset ──


class SmashClipDataset(Dataset):
    """Load pre-extracted features for SmashClip baselines.

    All features are loaded into memory at init time.

    Parameters
    ----------
    data_root : str or Path
        Path to the project root directory.
    split : str
        One of "train", "val", "test".
    task : str
        Task identifier (used for metadata handling).
    visual_dim : int
        Dimension of visual features (768 for InternVideo2-6B).
    visual_dir : str or None
        Override name for the visual feature directory.
    annotation_file : str or None
        Relative path from data_root to annotation JSONL file.
    """

    def __init__(
        self, data_root: str | Path, split: str, task: str, visual_dim: int = 768,
        visual_dir: str | None = None,
        annotation_file: str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.task = task.upper()
        self.visual_dim = visual_dim

        # Load split IDs
        split_path = self.data_root / "data" / "splits" / f"{split}.json"
        with open(split_path) as f:
            split_ids: list[str] = json.load(f)

        # Load annotation file
        self._annotations: Optional[dict[str, dict]] = None
        if annotation_file is not None:
            ann_path = self.data_root / annotation_file
            self._annotations = _load_annotations(ann_path)
            ann_ids = set(self._annotations.keys())
            self.ids = [cid for cid in split_ids if cid in ann_ids]
        else:
            self.ids = split_ids

        # Load vocab for metadata ID lookup
        vocab_path = self.data_root / "data" / "features" / "metadata" / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self._vocab = json.load(f)
        else:
            self._vocab = None

        # Pre-load all features
        feat_root = self.data_root / "data" / "features"
        iv2_dir = visual_dir if visual_dir is not None else "v1_video"

        self._visual: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []
        self._text: list[np.ndarray] = []
        self._prosody: list[np.ndarray] = []
        self._metadata: list[dict] = []

        for clip_id in self.ids:
            self._visual.append(
                np.load(feat_root / iv2_dir / f"{clip_id}.npy")
            )
            self._audio.append(
                np.load(feat_root / "a1_audio" / f"{clip_id}.npy")
            )
            self._text.append(
                np.load(feat_root / "a2_transcript" / f"{clip_id}.npy")
            )
            self._prosody.append(
                _load_prosody(feat_root / "a3_prosody" / f"{clip_id}.json")
            )
            if self._annotations is not None:
                ann = self._annotations[clip_id]
                meta = self._build_metadata_from_annotation(ann, feat_root, clip_id)
            else:
                with open(feat_root / "metadata" / f"{clip_id}.json") as f:
                    meta = json.load(f)
            self._metadata.append(meta)

    def _build_metadata_from_annotation(
        self, ann: dict, feat_root: Path, clip_id: str,
    ) -> dict:
        """Build metadata dict from annotation record + vocab."""
        vocab = self._vocab
        killer_id = vocab["characters"].get(ann["killer"], 0) if vocab else 0
        victim_id = vocab["characters"].get(ann["victim"], 0) if vocab else 0
        move_id = vocab["moves"].get(ann["move"], 0) if vocab else 0
        stage_id = vocab["stages"].get(ann["stage"], 0) if vocab else 0

        # Build scene_tags_vec (25-dim binary)
        scene_tags_vec = [0] * 25
        if vocab and "scene_tags" in ann:
            for tag in ann["scene_tags"]:
                idx = vocab["scene_tags"].get(tag)
                if idx is not None:
                    scene_tags_vec[idx] = 1

        return {
            "id": clip_id,
            "killer_id": killer_id,
            "victim_id": victim_id,
            "move_id": move_id,
            "stage_id": stage_id,
            "scene_tags_vec": scene_tags_vec,
            "score": float(ann["mean_score"]),
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        meta = self._metadata[idx]
        result = {
            "internvideo2": torch.from_numpy(self._visual[idx]),      # (T, D)
            "beats_audio": torch.from_numpy(self._audio[idx]),        # (768,)
            "text_embed": torch.from_numpy(self._text[idx]),          # (768,)
            "prosody": torch.from_numpy(self._prosody[idx]),          # (12,)
            "killer_id": meta["killer_id"],
            "victim_id": meta["victim_id"],
            "move_id": meta["move_id"],
            "stage_id": meta["stage_id"],
            "scene_tags": torch.tensor(meta["scene_tags_vec"], dtype=torch.float32),
            "clip_id": self.ids[idx],
        }
        if self._annotations is not None:
            result["score"] = meta["score"]
        else:
            result["score"] = meta["aesthetic_score"] - 1
        return result


# ── Collate ──


def collate_fn(batch: list[dict]) -> dict:
    """Collate with zero-padding for variable-length visual features."""
    max_t = max(item["internvideo2"].shape[0] for item in batch)
    feat_dim = batch[0]["internvideo2"].shape[1]

    iv2_padded = []
    iv2_mask = []
    for item in batch:
        t = item["internvideo2"].shape[0]
        pad = torch.zeros(max_t - t, feat_dim)
        iv2_padded.append(torch.cat([item["internvideo2"], pad], dim=0))
        mask = torch.zeros(max_t, dtype=torch.bool)
        mask[:t] = True
        iv2_mask.append(mask)

    return {
        "internvideo2": torch.stack(iv2_padded),       # (B, T, D)
        "internvideo2_mask": torch.stack(iv2_mask),    # (B, T) True=valid
        "beats_audio": torch.stack([b["beats_audio"] for b in batch]),
        "text_embed": torch.stack([b["text_embed"] for b in batch]),
        "prosody": torch.stack([b["prosody"] for b in batch]),
        "killer_id": torch.tensor([b["killer_id"] for b in batch], dtype=torch.long),
        "victim_id": torch.tensor([b["victim_id"] for b in batch], dtype=torch.long),
        "move_id": torch.tensor([b["move_id"] for b in batch], dtype=torch.long),
        "stage_id": torch.tensor([b["stage_id"] for b in batch], dtype=torch.long),
        "scene_tags": torch.stack([b["scene_tags"] for b in batch]),
        "score": torch.tensor(
            [b["score"] for b in batch], dtype=torch.float32
        ),
        "clip_id": [b["clip_id"] for b in batch],
    }
