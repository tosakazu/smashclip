"""Extract prosody features (pitch, energy, speech rate, spectral centroid).

Input: data/audio/{id}.wav (16kHz mono)
Output: data/features/a3_prosody/{id}.json -- 12-dim scalar features + timeseries

Requirements:
  pip install librosa

Usage:
  python feature_extraction/extract_prosody.py
  python feature_extraction/extract_prosody.py --workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts/en")
OUTPUT_DIR = Path("data/features/a3_prosody")

SAMPLE_RATE = 16000
FRAME_SEC = 0.5
HOP_LENGTH = 512
FMIN = 65.0
FMAX = 2093.0


def _pitched_frames(y: np.ndarray, sr: int) -> np.ndarray:
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, fmin=FMIN, fmax=FMAX, hop_length=HOP_LENGTH,
    )
    idx = magnitudes.argmax(axis=0)
    pitch_vals = pitches[idx, np.arange(pitches.shape[1])]
    return pitch_vals[pitch_vals > 0]


def _extract_pitch(y: np.ndarray, sr: int) -> dict:
    voiced = _pitched_frames(y, sr)
    if len(voiced) == 0:
        return {"mean": 0.0, "max": 0.0, "std": 0.0, "max_mean_ratio": 0.0}
    mean_val = float(np.mean(voiced))
    max_val = float(np.max(voiced))
    return {
        "mean": round(mean_val, 2), "max": round(max_val, 2),
        "std": round(float(np.std(voiced)), 2),
        "max_mean_ratio": round(max_val / mean_val, 2) if mean_val > 0 else 0.0,
    }


def _extract_energy(y: np.ndarray, duration: float) -> dict:
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    if len(rms) == 0:
        return {"mean": 0.0, "max": 0.0, "std": 0.0, "rate_of_change": 0.0}
    mean_val = float(np.mean(rms))
    max_val = float(np.max(rms))
    min_val = float(np.min(rms))
    roc = (max_val - min_val) / duration if duration > 0 else 0.0
    return {
        "mean": round(mean_val, 6), "max": round(max_val, 6),
        "std": round(float(np.std(rms)), 6), "rate_of_change": round(roc, 6),
    }


def _extract_speech_rate(transcript_path: Path) -> dict:
    default = {"chars_per_sec": 0.0, "words_per_sec": 0.0, "segment_count": 0}
    if not transcript_path.exists():
        return default
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        return default

    total_chars, total_words, total_duration = 0, 0, 0.0
    for seg in segments:
        text = seg.get("text", "").strip()
        seg_dur = seg["end"] - seg["start"]
        if seg_dur > 0 and text:
            total_chars += len(text)
            total_words += len(text.split())
            total_duration += seg_dur

    if total_duration == 0:
        return {**default, "segment_count": len(segments)}
    return {
        "chars_per_sec": round(total_chars / total_duration, 2),
        "words_per_sec": round(total_words / total_duration, 2),
        "segment_count": len(segments),
    }


def _extract_spectral_centroid(y: np.ndarray, sr: int) -> dict:
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    if len(sc) == 0:
        return {"mean": 0.0}
    return {"mean": round(float(np.mean(sc)), 2)}


def _extract_timeseries(y: np.ndarray, sr: int) -> dict:
    frame_samples = int(FRAME_SEC * sr)
    n_frames = max(1, int(np.ceil(len(y) / frame_samples)))
    pitch_ts, energy_ts, centroid_ts = [], [], []

    for i in range(n_frames):
        start = i * frame_samples
        chunk = y[start : min((i + 1) * frame_samples, len(y))]
        if len(chunk) == 0:
            pitch_ts.append(0.0); energy_ts.append(0.0); centroid_ts.append(0.0)
            continue
        voiced = _pitched_frames(chunk, sr)
        pitch_ts.append(round(float(np.mean(voiced)), 2) if len(voiced) > 0 else 0.0)
        rms = librosa.feature.rms(y=chunk, hop_length=HOP_LENGTH)[0]
        energy_ts.append(round(float(np.mean(rms)), 6) if len(rms) > 0 else 0.0)
        sc = librosa.feature.spectral_centroid(y=chunk, sr=sr, hop_length=HOP_LENGTH)[0]
        centroid_ts.append(round(float(np.mean(sc)), 2) if len(sc) > 0 else 0.0)

    return {"frame_sec": FRAME_SEC, "pitch": pitch_ts, "energy": energy_ts, "spectral_centroid": centroid_ts}


def process_one(audio_id: str) -> str | None:
    output_path = OUTPUT_DIR / f"{audio_id}.json"
    if output_path.exists():
        return None
    audio_path = AUDIO_DIR / f"{audio_id}.wav"
    if not audio_path.exists():
        return audio_id
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return audio_id

    duration = float(len(y) / sr)
    result = {
        "id": audio_id, "duration": round(duration, 3),
        "pitch": _extract_pitch(y, sr),
        "energy": _extract_energy(y, duration),
        "speech_rate": _extract_speech_rate(TRANSCRIPT_DIR / f"{audio_id}.json"),
        "spectral_centroid": _extract_spectral_centroid(y, sr),
        "timeseries": _extract_timeseries(y, sr),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    num_workers = args.workers or min(os.cpu_count() or 4, 16)

    audio_ids = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    existing = {p.stem for p in OUTPUT_DIR.glob("*.json")} if OUTPUT_DIR.exists() else set()
    remaining = [aid for aid in audio_ids if aid not in existing]

    print(f"Total: {len(audio_ids)}, Done: {len(existing)}, Remaining: {len(remaining)}")
    if not remaining:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(lambda x: process_one(x), remaining),
                           total=len(remaining), desc="Prosody"):
            if result is not None:
                errors.append(result)

    print(f"Done. Features: {len(list(OUTPUT_DIR.glob('*.json')))}/{len(audio_ids)}")
    if errors:
        print(f"Errors ({len(errors)}): {errors[:10]}")


if __name__ == "__main__":
    main()
