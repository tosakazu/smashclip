"""Training script for all SmashClip benchmark tasks.

Usage
-----
python scripts/train.py --task A                        # Task A: regression
python scripts/train.py --task Acls                     # Task A: 5-class classification
python scripts/train.py --task B                        # Task B: metadata prediction
python scripts/train.py --task C                        # Task C: scene tag prediction
python scripts/train.py --task A --models V1,A1         # Specific models only
python scripts/train.py --task B --targets killer,stage # Specific targets
python scripts/train.py --task A --test-run             # 1 seed, 3 epochs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import SmashClipDataset, collate_fn
from src.fusion import EarlyFusionModel
from scipy.stats import pearsonr, spearmanr
from src.metrics import (
    compute_mean_predictor_baseline,
    compute_random_baseline,
    compute_random_regression_baseline,
    compute_regression_metrics,
    compute_metadata_metrics,
    compute_task_a_metrics,
    compute_task_c_metrics,
    compute_majority_baseline,
)
from src.models import MetadataPredictionModel, build_model
from src.utils import EarlyStopping, save_results, set_seed

# ── Constants ──

DATA_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = "data/annotations/smashclip_en.jsonl"
RESULTS_DIR = DATA_ROOT / "results"

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCHS = 100
PATIENCE = 20

# ── Hyperparameter configs ──

UNIMODAL_HP = {
    "V1": {"lr": 5e-4, "wd": 0.01, "warmup": 0, "hidden": 256, "dropout": 0.1},
    "A1": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
    "A2": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
    "A3": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
}

FUSION_HP = {
    "V1+A1": {"lr": 1e-3, "wd": 0.01, "warmup": 5, "proj_dim": 256, "n_layers": 2, "head_hidden": 256, "head_dropout": 0.1},
    "V1+A2": {"lr": 1e-3, "wd": 0.01, "warmup": 5, "proj_dim": 256, "n_layers": 2, "head_hidden": 256, "head_dropout": 0.1},
    "V1+A3": {"lr": 1e-3, "wd": 0.01, "warmup": 5, "proj_dim": 256, "n_layers": 2, "head_hidden": 256, "head_dropout": 0.1},
    "A1+A2+A3": {"lr": 1e-3, "wd": 0.01, "warmup": 5, "proj_dim": 256, "n_layers": 2, "head_hidden": 256, "head_dropout": 0.1},
    "A1+A2+A3+V1": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
}

METADATA_HP = {"lr": 5e-4, "wd": 0.01, "warmup": 0, "hidden": 256, "dropout": 0.1}

CLS_UNIMODAL_HP = {
    "V1": {"lr": 5e-4, "wd": 1e-4, "warmup": 5, "hidden": 512, "dropout": 0.5},
    "A1": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
    "A2": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
    "A3": {"lr": 1e-2, "wd": 1e-4, "warmup": 0, "hidden": 256, "dropout": 0.3},
}

CLS_FUSION_HP = {
    "V1+A1": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
    "V1+A2": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
    "V1+A3": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
    "A1+A2+A3": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
    "A1+A2+A3+V1": {"lr": 5e-4, "wd": 0.001, "warmup": 5, "proj_dim": 128, "n_layers": 1, "head_hidden": 256, "head_dropout": 0.3},
}

# ── Task definitions ──

TASK_A_UNIMODAL = ["V1", "A1", "A2", "A3"]
TASK_A_FUSION = ["V1+A1", "V1+A2", "V1+A3", "A1+A2+A3", "A1+A2+A3+V1"]
TASK_A_BASELINES = ["mean_predictor", "random"]
TASK_A_ALL = TASK_A_UNIMODAL + TASK_A_FUSION + TASK_A_BASELINES

TASK_B_TARGETS = ["killer", "victim", "stage", "move"]
TASK_B_NUM_CLASSES = {"killer": 86, "victim": 86, "stage": 8, "move": 24}

TASK_C_UNIMODAL = ["V1", "A1", "A2", "A3"]
TASK_C_FUSION = ["A1+A2+A3+V1"]
TASK_C_BASELINES = ["random", "majority"]
TASK_C_ALL = TASK_C_UNIMODAL + TASK_C_FUSION + TASK_C_BASELINES


# ── Training helpers ──


def _build_scheduler(
    optimizer: torch.optim.Optimizer, hp: dict, epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine scheduler with optional linear warmup."""
    warmup = hp.get("warmup", 0)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    if warmup == 0:
        return cosine
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup],
    )


def _parse_modalities(model_name: str) -> list[str]:
    return model_name.split("+")


def _is_fusion(model_name: str) -> bool:
    return "+" in model_name


def _build_unimodal_model(
    model_name: str, output_dim: int, hp: dict, visual_dim: int = 768,
) -> nn.Module:
    return build_model(model_name, "A", output_dim, visual_dim=visual_dim)


def _build_fusion_model(
    model_name: str, task: str, output_dim: int, hp: dict, visual_dim: int = 768,
) -> nn.Module:
    modalities = _parse_modalities(model_name)
    return EarlyFusionModel(
        modalities=modalities, task=task, output_dim=output_dim,
        visual_dim=visual_dim, proj_dim=hp["proj_dim"],
        n_layers=hp["n_layers"], head_hidden=hp["head_hidden"],
        head_dropout=hp["head_dropout"],
    )


def _load_datasets(
    task_letter: str,
) -> tuple[SmashClipDataset, SmashClipDataset, SmashClipDataset]:
    ds_kwargs = dict(
        data_root=DATA_ROOT, task="A", visual_dim=768,
        annotation_file=ANNOTATION_FILE,
    )
    train_ds = SmashClipDataset(split="train", **ds_kwargs)
    val_ds = SmashClipDataset(split="val", **ds_kwargs)
    test_ds = SmashClipDataset(split="test", **ds_kwargs)
    return train_ds, val_ds, test_ds


def _make_loader(ds: SmashClipDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )


def _get_scores(ds: SmashClipDataset) -> np.ndarray:
    return np.array([ds[i]["score"] for i in range(len(ds))])


def _get_scene_tags(ds: SmashClipDataset) -> np.ndarray:
    return np.array([ds[i]["scene_tags"].numpy() for i in range(len(ds))])


# ── Task A: Regression ──


def _train_one_epoch_regression(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch).squeeze(-1)
        loss = criterion(logits, batch["score"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate_regression(
    model: nn.Module, loader: DataLoader, device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch).squeeze(-1)
        all_preds.append(logits.cpu().numpy())
        all_labels.append(batch["score"].cpu().numpy())
        if "clip_id" in batch:
            all_ids.extend(batch["clip_id"])
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_regression_metrics(preds, labels)
    metrics["pm1_agreement"] = round(float(np.mean(np.abs(preds - labels) <= 1.0)), 4)
    if return_predictions:
        return metrics, preds, labels, all_ids
    return metrics


def train_task_a_single_seed(
    model_name: str, seed: int,
    train_ds: SmashClipDataset, val_ds: SmashClipDataset, test_ds: SmashClipDataset,
    device: torch.device, epochs: int = EPOCHS,
) -> dict:
    set_seed(seed)

    if _is_fusion(model_name):
        hp = FUSION_HP[model_name]
        model = _build_fusion_model(model_name, "A", 1, hp).to(device)
    else:
        hp = UNIMODAL_HP[model_name]
        model = _build_unimodal_model(model_name, 1, hp).to(device)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = _build_scheduler(optimizer, hp, epochs)
    es = EarlyStopping(patience=PATIENCE, mode="max")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch_regression(model, train_loader, criterion, optimizer, device)
        val_metrics = _evaluate_regression(model, val_loader, device)
        scheduler.step()

        es.step(val_metrics["srcc"])
        if es.is_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        elapsed = time.time() - t0
        print(
            f"  seed={seed} epoch={epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} val_srcc={val_metrics['srcc']:.4f} "
            f"{'*' if es.is_best else ' '} ({elapsed:.1f}s)"
        )
        if es.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics, test_preds, test_labels, test_ids = _evaluate_regression(
        model, test_loader, device, return_predictions=True
    )
    return {
        "seed": seed, "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }


def run_task_a(
    model_name: str, seeds: list[int], device: torch.device, epochs: int = EPOCHS,
) -> None:
    print(f"\n{'='*60}")
    print(f"Task A (regression) | Model: {model_name}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = _load_datasets("A")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if model_name == "mean_predictor":
        train_scores = _get_scores(train_ds)
        test_scores = _get_scores(test_ds)
        metrics = compute_mean_predictor_baseline(train_scores, test_scores)
        result = {"model": model_name, "task": "A", "summary": metrics}
    elif model_name == "random":
        train_scores = _get_scores(train_ds)
        test_scores = _get_scores(test_ds)
        metrics = compute_random_regression_baseline(train_scores, test_scores)
        result = {"model": model_name, "task": "A", "summary": metrics}
    else:
        per_seed = []
        for seed in tqdm(seeds, desc=f"Task A {model_name}"):
            seed_result = train_task_a_single_seed(
                model_name, seed, train_ds, val_ds, test_ds, device, epochs,
            )
            per_seed.append(seed_result)

        metric_keys = per_seed[0]["test_metrics"].keys()
        summary = {}
        for k in metric_keys:
            values = [s["test_metrics"][k] for s in per_seed]
            summary[k] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

        hp = FUSION_HP[model_name] if _is_fusion(model_name) else UNIMODAL_HP[model_name]
        result = {
            "model": model_name, "task": "A",
            "config": {"batch_size": BATCH_SIZE, "epochs": epochs, "patience": PATIENCE, **hp},
            "per_seed": per_seed, "summary": summary,
        }

    out_path = RESULTS_DIR / f"A_{model_name}.json"
    save_results(result, out_path)
    print(f"Saved: {out_path}")
    print(f"Summary: {result['summary']}")


# ── Task B: Metadata classification ──


def _train_one_epoch_classification(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, target_key: str, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch["internvideo2"], batch["internvideo2_mask"])
        loss = criterion(logits, batch[f"{target_key}_id"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate_classification(
    model: nn.Module, loader: DataLoader, target_key: str, device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch["internvideo2"], batch["internvideo2_mask"])
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch[f"{target_key}_id"].cpu().numpy())
        if "clip_id" in batch:
            all_ids.extend(batch["clip_id"])
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    metrics = compute_metadata_metrics(preds_np, labels_np)
    if return_predictions:
        return metrics, preds_np, labels_np, all_ids
    return metrics


def train_task_b_single_seed(
    target: str, seed: int,
    train_ds: SmashClipDataset, val_ds: SmashClipDataset, test_ds: SmashClipDataset,
    device: torch.device, epochs: int = EPOCHS,
) -> dict:
    set_seed(seed)
    hp = METADATA_HP
    num_classes = TASK_B_NUM_CLASSES[target]
    model = MetadataPredictionModel(num_classes=num_classes, visual_dim=768).to(device)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = _build_scheduler(optimizer, hp, epochs)
    es = EarlyStopping(patience=PATIENCE, mode="max")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch_classification(
            model, train_loader, criterion, optimizer, target, device,
        )
        val_metrics = _evaluate_classification(model, val_loader, target, device)
        scheduler.step()

        es.step(val_metrics["balanced_accuracy"])
        if es.is_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        elapsed = time.time() - t0
        print(
            f"  seed={seed} epoch={epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"{'*' if es.is_best else ' '} ({elapsed:.1f}s)"
        )
        if es.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate_classification(model, test_loader, target, device)
    return {"seed": seed, "best_epoch": best_epoch, "test_metrics": test_metrics}


def run_task_b(
    target: str, seeds: list[int], device: torch.device, epochs: int = EPOCHS,
) -> None:
    print(f"\n{'='*60}")
    print(f"Task B (metadata) | Target: {target}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = _load_datasets("B")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    per_seed = []
    for seed in tqdm(seeds, desc=f"Task B {target}"):
        seed_result = train_task_b_single_seed(
            target, seed, train_ds, val_ds, test_ds, device, epochs,
        )
        per_seed.append(seed_result)

    metric_keys = per_seed[0]["test_metrics"].keys()
    summary = {}
    for k in metric_keys:
        values = [s["test_metrics"][k] for s in per_seed]
        summary[k] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

    result = {
        "model": f"MetadataPrediction_{target}", "task": "B", "target": target,
        "config": {"batch_size": BATCH_SIZE, "epochs": epochs, "patience": PATIENCE, **METADATA_HP},
        "per_seed": per_seed, "summary": summary,
    }
    out_path = RESULTS_DIR / f"B_{target}.json"
    save_results(result, out_path)
    print(f"Saved: {out_path}")
    print(f"Summary: {result['summary']}")


# ── Task C: Multi-label scene tags ──


def _train_one_epoch_multilabel(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        loss = criterion(logits, batch["scene_tags"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate_multilabel(
    model: nn.Module, loader: DataLoader, device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["scene_tags"].cpu().numpy())
        if "clip_id" in batch:
            all_ids.extend(batch["clip_id"])
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    metrics = compute_task_c_metrics(logits_np, labels_np)
    if return_predictions:
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        preds = (probs >= 0.5).astype(int)
        return metrics, preds, labels_np, all_ids
    return metrics


def train_task_c_single_seed(
    model_name: str, seed: int,
    train_ds: SmashClipDataset, val_ds: SmashClipDataset, test_ds: SmashClipDataset,
    device: torch.device, epochs: int = EPOCHS,
) -> dict:
    set_seed(seed)
    output_dim = 25

    if _is_fusion(model_name):
        hp = FUSION_HP[model_name]
        model = _build_fusion_model(model_name, "B", output_dim, hp).to(device)
    else:
        hp = UNIMODAL_HP[model_name]
        model = _build_unimodal_model(model_name, output_dim, hp).to(device)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = _build_scheduler(optimizer, hp, epochs)
    es = EarlyStopping(patience=PATIENCE, mode="max")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch_multilabel(model, train_loader, criterion, optimizer, device)
        val_metrics = _evaluate_multilabel(model, val_loader, device)
        scheduler.step()

        es.step(val_metrics["mAP"])
        if es.is_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        elapsed = time.time() - t0
        print(
            f"  seed={seed} epoch={epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} val_mAP={val_metrics['mAP']:.4f} "
            f"{'*' if es.is_best else ' '} ({elapsed:.1f}s)"
        )
        if es.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate_multilabel(model, test_loader, device)
    return {"seed": seed, "best_epoch": best_epoch, "test_metrics": test_metrics}


def run_task_c(
    model_name: str, seeds: list[int], device: torch.device, epochs: int = EPOCHS,
) -> None:
    print(f"\n{'='*60}")
    print(f"Task C (scene tags) | Model: {model_name}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = _load_datasets("C")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if model_name == "random":
        test_labels = _get_scene_tags(test_ds)
        metrics = compute_random_baseline(test_labels, "C")
        result = {"model": model_name, "task": "C", "summary": metrics}
    elif model_name == "majority":
        train_labels = _get_scene_tags(train_ds)
        test_labels = _get_scene_tags(test_ds)
        metrics = compute_majority_baseline(train_labels, test_labels, "C")
        result = {"model": model_name, "task": "C", "summary": metrics}
    else:
        per_seed = []
        for seed in tqdm(seeds, desc=f"Task C {model_name}"):
            seed_result = train_task_c_single_seed(
                model_name, seed, train_ds, val_ds, test_ds, device, epochs,
            )
            per_seed.append(seed_result)

        metric_keys = per_seed[0]["test_metrics"].keys()
        summary = {}
        for k in metric_keys:
            values = [s["test_metrics"][k] for s in per_seed]
            summary[k] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

        hp = FUSION_HP[model_name] if _is_fusion(model_name) else UNIMODAL_HP[model_name]
        result = {
            "model": model_name, "task": "C",
            "config": {"batch_size": BATCH_SIZE, "epochs": epochs, "patience": PATIENCE, **hp},
            "per_seed": per_seed, "summary": summary,
        }

    out_path = RESULTS_DIR / f"C_{model_name}.json"
    save_results(result, out_path)
    print(f"Saved: {out_path}")
    print(f"Summary: {result['summary']}")


# ── Task Acls: 5-class classification ──


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _build_cls_criterion(loss_type: str, train_ds: SmashClipDataset, device: torch.device) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    scores = np.array([train_ds[i]["score"] for i in range(len(train_ds))])
    classes = np.round(scores).clip(1, 5).astype(int) - 1
    counts = np.bincount(classes, minlength=5).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 5
    weight_tensor = torch.FloatTensor(weights).to(device)
    if loss_type == "wce":
        return nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_type == "focal":
        return FocalLoss(alpha=weight_tensor, gamma=2.0)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _train_one_epoch_cls(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        target = batch["score"].round().clamp(1, 5).long() - 1
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate_cls(
    model: nn.Module, loader: DataLoader, device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        preds = logits.argmax(dim=-1)
        target = batch["score"].round().clamp(1, 5).long() - 1
        all_preds.append(preds.cpu().numpy())
        all_labels.append(target.cpu().numpy())
        if "clip_id" in batch:
            all_ids.extend(batch["clip_id"])
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    logits_onehot = np.full((len(preds_np), 5), -10.0)
    logits_onehot[np.arange(len(preds_np)), preds_np] = 10.0
    metrics = compute_task_a_metrics(logits_onehot, labels_np)
    preds_15 = preds_np.astype(float) + 1
    labels_15 = labels_np.astype(float) + 1
    srcc, _ = spearmanr(labels_15, preds_15)
    plcc, _ = pearsonr(labels_15, preds_15)
    mae = float(np.mean(np.abs(preds_15 - labels_15)))
    pm1 = float(np.mean(np.abs(preds_15 - labels_15) <= 1))
    metrics["srcc"] = round(float(srcc), 4)
    metrics["plcc"] = round(float(plcc), 4)
    metrics["mae"] = round(float(mae), 4)
    metrics["pm1_agreement"] = round(pm1, 4)
    if return_predictions:
        return metrics, preds_15, labels_15, all_ids
    return metrics


def train_task_acls_single_seed(
    model_name: str, seed: int,
    train_ds: SmashClipDataset, val_ds: SmashClipDataset, test_ds: SmashClipDataset,
    device: torch.device, epochs: int = EPOCHS, loss_type: str = "ce",
) -> dict:
    set_seed(seed)

    if _is_fusion(model_name):
        hp = CLS_FUSION_HP[model_name]
        model = _build_fusion_model(model_name, "A", 5, hp).to(device)
    else:
        hp = CLS_UNIMODAL_HP[model_name]
        model = _build_unimodal_model(model_name, 5, hp).to(device)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    criterion = _build_cls_criterion(loss_type, train_ds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = _build_scheduler(optimizer, hp, epochs)
    es = EarlyStopping(patience=PATIENCE, mode="max")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch_cls(model, train_loader, criterion, optimizer, device)
        val_metrics = _evaluate_cls(model, val_loader, device)
        scheduler.step()

        es.step(val_metrics["weighted_f1"])
        if es.is_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        elapsed = time.time() - t0
        print(
            f"  seed={seed} epoch={epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} val_wf1={val_metrics['weighted_f1']:.4f} "
            f"{'*' if es.is_best else ' '} ({elapsed:.1f}s)"
        )
        if es.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate_cls(model, test_loader, device)
    return {"seed": seed, "best_epoch": best_epoch, "test_metrics": test_metrics}


def run_task_acls(
    model_name: str, seeds: list[int], device: torch.device,
    epochs: int = EPOCHS, loss_type: str = "ce",
) -> None:
    print(f"\n{'='*60}")
    print(f"Task Acls (5-class, loss={loss_type}) | Model: {model_name}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = _load_datasets("A")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if model_name in ("mean_predictor", "random"):
        train_labels = np.round(_get_scores(train_ds)).clip(1, 5).astype(int) - 1
        test_labels = np.round(_get_scores(test_ds)).clip(1, 5).astype(int) - 1
        if model_name == "mean_predictor":
            from scipy.stats import mode
            majority_class = int(mode(train_labels, keepdims=False).mode)
            logits = np.full((len(test_labels), 5), -10.0)
            logits[:, majority_class] = 10.0
            metrics = compute_task_a_metrics(logits, test_labels)
        else:
            metrics = compute_random_baseline(test_labels, "A")
        result = {"model": model_name, "task": "Acls", "summary": metrics}
    else:
        per_seed = []
        for seed in tqdm(seeds, desc=f"Task Acls {model_name}"):
            seed_result = train_task_acls_single_seed(
                model_name, seed, train_ds, val_ds, test_ds, device, epochs,
                loss_type=loss_type,
            )
            per_seed.append(seed_result)

        metric_keys = per_seed[0]["test_metrics"].keys()
        summary = {}
        for k in metric_keys:
            values = [s["test_metrics"][k] for s in per_seed]
            summary[k] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

        hp = CLS_FUSION_HP[model_name] if _is_fusion(model_name) else CLS_UNIMODAL_HP[model_name]
        result = {
            "model": model_name, "task": "Acls",
            "config": {"batch_size": BATCH_SIZE, "epochs": epochs, "patience": PATIENCE, **hp},
            "per_seed": per_seed, "summary": summary,
        }

    out_path = RESULTS_DIR / f"Acls_{model_name}.json"
    save_results(result, out_path)
    print(f"Saved: {out_path}")
    print(f"Summary: {result['summary']}")


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(description="SmashClip training script")
    parser.add_argument("--task", type=str, required=True,
                        choices=["A", "Acls", "Acls_focal", "Acls_wce", "B", "C"],
                        help="Task: A (regression), Acls (5-class CE), Acls_focal, Acls_wce, B, C")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names for Task A/C (e.g. V1,A1,V1+A1)")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated targets for Task B (e.g. killer,stage)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--test-run", action="store_true",
                        help="Quick test: 1 seed, 3 epochs")
    args = parser.parse_args()

    if args.test_run:
        seeds = [42]
        epochs = 3
    else:
        seeds = args.seeds
        epochs = EPOCHS

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds} | Epochs: {epochs}")

    if args.task == "A":
        models = args.models.split(",") if args.models else TASK_A_ALL
        for model_name in models:
            run_task_a(model_name, seeds, device, epochs)
    elif args.task in ("Acls", "Acls_focal", "Acls_wce"):
        loss_map = {"Acls": "ce", "Acls_focal": "focal", "Acls_wce": "wce"}
        loss_type = loss_map[args.task]
        models = args.models.split(",") if args.models else TASK_A_ALL
        for model_name in models:
            run_task_acls(model_name, seeds, device, epochs, loss_type=loss_type)
    elif args.task == "B":
        targets = args.targets.split(",") if args.targets else TASK_B_TARGETS
        for target in targets:
            run_task_b(target, seeds, device, epochs)
    elif args.task == "C":
        models = args.models.split(",") if args.models else TASK_C_ALL
        for model_name in models:
            run_task_c(model_name, seeds, device, epochs)

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
