# SmashClip

A multimodal dataset for clip-worthiness assessment in competitive Super Smash Bros. Ultimate. SmashClip contains 2,503 KO clips annotated with clip-worthiness scores (1-5), scene tags (20 categories), metadata (characters, stage, finishing move), and English commentary transcripts. The dataset supports four benchmark tasks: clip-worthiness prediction, metadata prediction, scene tag prediction, and rationale generation.

**Paper:** Kazuhiro Saito and Yuchi Ishikawa. "SmashClip: A Multimodal Dataset for Clip-Worthiness Assessment of Competitive Game Videos." Under review.

**License:** Code: [MIT](LICENSE) | Data: [CC BY 4.0](LICENSE-DATA)

## Dataset Overview

| Property | Value |
|----------|-------|
| Total clips | 2,503 |
| Train / Val / Test | 1,771 / 341 / 391 |
| Annotators | 8 (3 per clip) |
| Score range | 1-5 (mean of 3 ratings) |
| Characters | 86 |
| Scene tags | 20 |
| Languages | English annotations, bilingual transcripts |
| Video sources | 92 |
| Stages | 8 |
| Move categories | 24 |

## Directory Structure

```
smashclip/
  data/
    annotations/
      smashclip_en.jsonl        # Main annotation file
      smashclip_ja.jsonl        # Japanese annotation file
    features/
      v1_video/                 # InternVideo2-6B visual features (T, 768)
      a1_audio/                 # BEATs audio features (768,)
      a2_transcript/            # MPNet transcript embeddings (768,)
      a3_prosody/               # Prosody features (JSON, 12-dim scalar)
      metadata/
        vocab.json              # Label vocabularies (characters, stages, moves, tags)
    splits/
      train.json, val.json, test.json
      rationale_train.json, rationale_val.json, rationale_test.json
    transcripts/
      en/                       # English transcripts (JSON, one per clip)
      original/                 # Original language transcripts (JSON)
  src/                          # Model and dataset code
  scripts/
    train.py                    # Training script
  feature_extraction/           # Feature extraction scripts
  docs/                         # Documentation
```

## Quick Start

### Install

```bash
pip install -e .
```

For feature extraction from raw video/audio:
```bash
pip install -e ".[features]"
```

### Train Baselines

```bash
# Task A: clip-worthiness prediction (all models, 5 seeds)
python scripts/train.py --task Acls

# Task A: single model, quick test
python scripts/train.py --task Acls --models V1 --test-run

# Task B: metadata prediction from visual features
python scripts/train.py --task B

# Task C: scene tag prediction (multi-label)
python scripts/train.py --task C
```

## Data Format

Each line in `smashclip_en.jsonl` is a JSON object:

```json
{
  "id": "45e47594-424c-492e-ba57-9670e1f07ccf",
  "video_url": "https://youtu.be/...",
  "title": "Tournament Name - Player1 Vs. Player2",
  "start_time": "00:04:07.305",
  "end_time": "00:04:14.546",
  "stage": "Town and City",
  "killer": "Peach",
  "victim": "Palutena",
  "move": "Bair",
  "scene_tags": ["Edgeguarding"],
  "caption": null,
  "players": ["Samsora(Peach)", "Dabuz(Palutena, Rosalina & Luma)"],
  "scores": {"E": 2, "F": 3, "A": 3},
  "mean_score": 2.67
}
```

500 clips in the rationale subset have a `caption` field with a free-text rationale explaining the clip-worthiness score; the remaining entries have `caption: null`.

## Benchmark Tasks

### Task A: Clip-Worthiness Prediction

Predict the integer-rounded mean of three annotator ratings (1-5). The model uses cross-entropy loss on integer labels. We evaluate with Accuracy, Balanced Accuracy, SRCC, and PLCC.

- **Classification:** Cross-entropy on integer-rounded labels (1-5 mapped to classes 0-4).
- **Regression (alternative):** MSE loss, output dim = 1. Available via `--task A`.

### Task B: Metadata Prediction

Predict game metadata (killer character, victim character, stage, finishing move) from visual features alone. Primary metric: balanced accuracy.

### Task C: Scene Tag Prediction

Multi-label classification over 20 scene tags describing tactical context and presentation elements. Primary metric: mAP. Loss: BCEWithLogits.

### Task D: Rationale Generation

Generate natural language explanations for why a clip received its clip-worthiness score. Evaluated with BERTScore, ROUGE-L, and Claude-as-judge (relevance, specificity, accuracy on 1-5 scales).

## Models

### Unimodal

| ID | Modality | Input | Architecture |
|----|----------|-------|-------------|
| V1 | Video | InternVideo2-6B (T, 768) | 1-layer Transformer + mean pool |
| V2 | Video | InternVideo2-6B (T, 768) | Mean pool (no Transformer) |
| A1 | Audio | BEATs (768) | MLP |
| A2 | Transcript | MPNet (768) | MLP |
| A3 | Prosody | 12-dim scalar | BatchNorm + MLP |
| M1 | Metadata | Categorical embeddings | MLP |

### Fusion (Early Fusion, F2)

Shared Transformer with modality tag embeddings and CLS token. Tested combinations: V1+A1, V1+A2, V1+A3, A1+A2+A3, A1+A2+A3+V1.

## Ethics and Data Collection

SmashClip is built from publicly available YouTube broadcasts of community-organized Super Smash Bros. Ultimate tournaments. The released data contains **annotations and pre-extracted features only**; no raw video or audio is redistributed. YouTube URLs point to videos hosted by their original uploaders.

- **Annotators:** Eight expert annotators participated. All were informed of the task and intended use, and provided consent. Annotators were compensated for their work.
- **Player information:** Player names in the dataset are public competitive gamer tags from tournament brackets and broadcasts. No real names, contact information, or other personal data is included. Player faces in figures are blurred.
- **Transcripts:** Commentary transcripts are short fragments (typically 5-20 seconds per clip) of live tournament broadcasts, included for research purposes. They represent a small, non-substitutive portion of the original broadcasts.
- **Pre-extracted features:** All distributed features (InternVideo2, BEATs, MPNet embeddings, prosody statistics) are non-reversible; original video and audio cannot be reconstructed from them.

## Takedown Policy

We respect the rights of players, commentators, and video uploaders. If you would like any clip referencing your content removed from the dataset, please open a GitHub issue or contact us at smash.tosakazu@gmail.com. We will honor takedown requests promptly.

## Copyright Notice

Super Smash Bros. Ultimate is a registered trademark of Nintendo. Game content, character names, and stage names are the property of Nintendo / HAL Laboratory, Inc. This dataset is an independent research project and is not affiliated with, endorsed by, or sponsored by Nintendo.

## License

- **Code:** MIT License (see `LICENSE`)
- **Data:** CC BY 4.0 (see `LICENSE-DATA`)

## Citation

```bibtex
@article{saito2026smashclip,
  title   = {SmashClip: A Multimodal Dataset for Clip-Worthiness Assessment of Competitive Game Videos},
  author  = {Saito, Kazuhiro and Ishikawa, Yuchi},
  year    = {2026},
}
```
