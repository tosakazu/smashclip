# Training Details

## General Configuration

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 123, 456, 789, 1024 |
| Batch size | 256 |
| Max epochs | 100 |
| Early stopping patience | 20 epochs |
| Optimizer | AdamW |
| Scheduler | Cosine annealing (with optional linear warmup) |
| Visual feature dim | 768 (InternVideo2-6B) |

## Loss Functions per Task

| Task | Loss | Monitor metric (early stopping) |
|------|------|---------------------------------|
| A (classification) | Cross-entropy | Validation weighted F1 |
| A (5-class) | Cross-entropy | Validation weighted F1 |
| A (5-class, focal) | Focal loss (gamma=2) | Validation weighted F1 |
| A (5-class, WCE) | Weighted cross-entropy | Validation weighted F1 |
| B (metadata) | Cross-entropy | Validation balanced accuracy |
| C (scene tags) | BCEWithLogits | Validation mAP |

## Unimodal Hyperparameters (Task A, Clip-Worthiness)

| Model | LR | Weight Decay | Warmup | Hidden | Dropout |
|-------|----|-------------|--------|--------|---------|
| V1 | 5e-4 | 0.01 | 0 | 256 | 0.1 |
| A1 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |
| A2 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |
| A3 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |

## Classification Unimodal Hyperparameters (Task A 5-class)

| Model | LR | Weight Decay | Warmup | Hidden | Dropout |
|-------|----|-------------|--------|--------|---------|
| V1 | 5e-4 | 1e-4 | 5 | 512 | 0.5 |
| A1 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |
| A2 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |
| A3 | 1e-2 | 1e-4 | 0 | 256 | 0.3 |

## Fusion Hyperparameters (Task A, Clip-Worthiness, Early Fusion F2)

| Modalities | LR | Weight Decay | Warmup | Proj Dim | Layers | Head Hidden | Head Dropout |
|------------|----|-------------|--------|----------|--------|-------------|--------------|
| V1+A1 | 1e-3 | 0.01 | 5 | 256 | 2 | 256 | 0.1 |
| V1+A2 | 1e-3 | 0.01 | 5 | 256 | 2 | 256 | 0.1 |
| V1+A3 | 1e-3 | 0.01 | 5 | 256 | 2 | 256 | 0.1 |
| A1+A2+A3 | 1e-3 | 0.01 | 5 | 256 | 2 | 256 | 0.1 |
| A1+A2+A3+V1 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |

## Classification Fusion Hyperparameters (Task A 5-class, Early Fusion F2)

| Modalities | LR | Weight Decay | Warmup | Proj Dim | Layers | Head Hidden | Head Dropout |
|------------|----|-------------|--------|----------|--------|-------------|--------------|
| V1+A1 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |
| V1+A2 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |
| V1+A3 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |
| A1+A2+A3 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |
| A1+A2+A3+V1 | 5e-4 | 0.001 | 5 | 128 | 1 | 256 | 0.3 |

## Metadata Prediction Hyperparameters (Task B)

| Parameter | Value |
|-----------|-------|
| LR | 5e-4 |
| Weight Decay | 0.01 |
| Warmup | 0 |
| Hidden | 256 |
| Dropout | 0.1 |
| Architecture | 1-layer Transformer + CLS token |

## Architecture Notes

- **Unimodal models (V1, V2, A1, A2, A3, M1):** Two-layer MLP head on top of feature encoder.
- **V1 (VisualTemporal):** 1-layer Transformer encoder with sinusoidal positional encoding, followed by masked mean pooling. Deeper Transformers (2 or 4 layers) showed overfitting.
- **V2 (VisualMeanPool):** Direct masked mean pooling without Transformer, used as ablation baseline.
- **A3 (Prosody):** BatchNorm before MLP head (12-dim input).
- **M1 (Metadata):** Learned 32-dim embeddings for killer, victim, move, and stage, concatenated (128-dim total). For Task A, 20-dim scene tag vector is also concatenated.
- **F2 (Early Fusion):** Shared Transformer encoder with learnable modality tag embeddings and a CLS token. Each modality is projected to a common dimension, tagged, and processed jointly.
- **Task B (MetadataPrediction):** 1-layer Transformer with CLS token, predicts metadata categories from visual features only.
