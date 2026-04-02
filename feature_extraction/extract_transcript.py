"""Extract transcript embeddings using all-mpnet-base-v2.

Model: sentence-transformers/all-mpnet-base-v2 (768-dim output)
Input: data/transcripts/en/{id}.json (English transcripts)
Output: data/features/a2_transcript/{id}.npy -- shape [768]

Requirements:
  pip install transformers

Usage:
  python feature_extraction/extract_transcript.py
  python feature_extraction/extract_transcript.py --model-path /path/to/all-mpnet-base-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

TRANSCRIPT_DIR = Path("data/transcripts/en")
OUTPUT_DIR = Path("data/features/a2_transcript")
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 256


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask (matches sentence-transformers behavior)."""
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def encode_texts(
    texts: list[str], tokenizer, model, device: str, batch_size: int,
) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=384, return_tensors="pt"
        ).to(device)
        outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded["attention_mask"])
        embeddings = normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                        help="Path or HuggingFace name of the sentence encoder")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device).eval()

    transcript_paths = sorted(TRANSCRIPT_DIR.glob("*.json"))
    existing = {p.stem for p in OUTPUT_DIR.glob("*.npy")} if OUTPUT_DIR.exists() else set()
    remaining = [p for p in transcript_paths if p.stem not in existing]

    print(f"Total: {len(transcript_paths)}, Done: {len(existing)}, Remaining: {len(remaining)}")
    if not remaining:
        return

    ids, texts, empty_ids = [], [], []
    for path in remaining:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        clip_id = data["id"]
        full_text = data.get("full_text", "").strip()
        if full_text:
            ids.append(clip_id)
            texts.append(full_text)
        else:
            empty_ids.append(clip_id)

    for clip_id in empty_ids:
        np.save(OUTPUT_DIR / f"{clip_id}.npy", np.zeros(EMBED_DIM, dtype=np.float32))

    if texts:
        embeddings = encode_texts(texts, tokenizer, model, device, BATCH_SIZE)
        for i, clip_id in enumerate(ids):
            np.save(OUTPUT_DIR / f"{clip_id}.npy", embeddings[i].astype(np.float32))

    print(f"Done. Embeddings: {len(list(OUTPUT_DIR.glob('*.npy')))}/{len(transcript_paths)}")


if __name__ == "__main__":
    main()
