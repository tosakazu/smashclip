"""Extract audio features using BEATs (Audio Pre-Training with Acoustic Tokenizers).

Model: Microsoft BEATs_iter3+ AS2M (768-dim output)
Input: data/audio/{id}.wav (16kHz mono)
Output: data/features/a1_audio/{id}.npy -- shape [768]

Requirements:
  - BEATs model code (see https://github.com/microsoft/unilm/tree/master/beats)
  - Checkpoint: BEATs_iter3_plus_AS2M.pt
  - pip install torchaudio

Usage:
  python feature_extraction/extract_audio.py --model-dir /path/to/beats
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

AUDIO_DIR = Path("data/audio")
OUTPUT_DIR = Path("data/features/a1_audio")

SAMPLE_RATE = 16000
MAX_CHUNK_SEC = 20


def _gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
        )
        return len(out.strip().splitlines())
    except Exception:
        return 0


def _build_model(model_dir: Path, device: torch.device):
    sys.path.insert(0, str(model_dir.resolve()))
    from BEATs import BEATs, BEATsConfig

    checkpoint_path = model_dir / "BEATs_iter3_plus_AS2M.pt"
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


def _extract_one(model, audio_path: Path, device: torch.device) -> np.ndarray | None:
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] == 0:
        return None

    chunk_samples = MAX_CHUNK_SEC * SAMPLE_RATE
    all_features = []
    with torch.no_grad():
        for start in range(0, waveform.shape[1], chunk_samples):
            chunk = waveform[:, start : start + chunk_samples].to(device)
            padding_mask = torch.zeros(chunk.shape, device=device).bool()
            features, _ = model.extract_features(chunk, padding_mask=padding_mask)
            all_features.append(features.mean(dim=1))

    stacked = torch.cat(all_features, dim=0)
    return stacked.mean(dim=0).cpu().float().numpy()


def extract_worker(args: tuple) -> list[str]:
    gpu_id, audio_ids, model_dir = args
    device = torch.device(f"cuda:{gpu_id}")
    model = _build_model(model_dir, device)
    errors = []

    for aid in tqdm(audio_ids, desc=f"GPU {gpu_id}", position=gpu_id):
        output_path = OUTPUT_DIR / f"{aid}.npy"
        if output_path.exists():
            continue
        audio_path = AUDIO_DIR / f"{aid}.wav"
        if not audio_path.exists():
            errors.append(aid)
            continue
        try:
            feat = _extract_one(model, audio_path, device)
            if feat is None or np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                errors.append(aid)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, feat)
        except Exception as e:
            tqdm.write(f"Error {aid}: {e}")
            errors.append(aid)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to BEATs model directory")
    args = parser.parse_args()

    num_gpus = _gpu_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    audio_ids = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    existing = {p.stem for p in OUTPUT_DIR.glob("*.npy")} if OUTPUT_DIR.exists() else set()
    remaining = [aid for aid in audio_ids if aid not in existing]

    print(f"Total: {len(audio_ids)}, Done: {len(existing)}, Remaining: {len(remaining)}")
    if not remaining:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks: list[list[str]] = [[] for _ in range(num_gpus)]
    for i, aid in enumerate(remaining):
        chunks[i % num_gpus].append(aid)

    model_dir = Path(args.model_dir)
    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_gpus) as pool:
        results = pool.map(extract_worker, [(i, c, model_dir) for i, c in enumerate(chunks)])

    all_errors = [e for errs in results for e in errs]
    done = len(list(OUTPUT_DIR.glob("*.npy")))
    print(f"Done. Features: {done}/{len(audio_ids)}")
    if all_errors:
        print(f"Errors ({len(all_errors)}): {all_errors[:20]}")


if __name__ == "__main__":
    main()
