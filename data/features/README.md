# Feature Descriptions

Pre-extracted features for all 2,503 clips. Each clip has a UUID as its filename.

## v1_video/ -- Visual Features

- **Model:** InternVideo2-Stage2_6B (vision encoder only)
- **Format:** `{id}.npy`, NumPy float32 array
- **Shape:** `(T, 768)` where T is the number of 4-frame segments (varies per clip, typically 15-45)
- **Extraction:** Video decoded at 10 FPS, grouped into 4-frame segments, each segment encoded to a 768-dim vector via the InternVideo2 vision encoder (CLIP projection head output)

## a1_audio/ -- Audio Features

- **Model:** BEATs_iter3+ AS2M (Microsoft)
- **Format:** `{id}.npy`, NumPy float32 array
- **Shape:** `(768,)` -- single vector per clip
- **Extraction:** Audio loaded at 16kHz mono, split into 20-second chunks if needed, each chunk encoded by BEATs, then mean-pooled across time and chunks

## a2_transcript/ -- Transcript Embeddings

- **Model:** all-mpnet-base-v2 (sentence-transformers)
- **Format:** `{id}.npy`, NumPy float32 array
- **Shape:** `(768,)` -- single L2-normalized vector per clip
- **Extraction:** English commentary transcript encoded with mean pooling over token embeddings. Clips with no commentary get a zero vector.

## a3_prosody/ -- Prosody Features

- **Extraction:** librosa (pitch via piptrack, RMS energy, spectral centroid) + Whisper transcript timing
- **Format:** `{id}.json`, JSON file
- **Scalar features (12-dim):**
  - `pitch.mean`, `pitch.max`, `pitch.std`, `pitch.max_mean_ratio`
  - `energy.mean`, `energy.max`, `energy.std`, `energy.rate_of_change`
  - `speech_rate.chars_per_sec`, `speech_rate.words_per_sec`, `speech_rate.segment_count`
  - `spectral_centroid.mean`
- **Timeseries:** Per-frame (0.5s windows) pitch, energy, and spectral centroid values
- **Duration:** Clip duration in seconds

## File Counts

All directories contain exactly 2,503 files, one per clip.
