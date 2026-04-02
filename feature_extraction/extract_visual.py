"""Extract visual features from video clips using InternVideo2-Stage2_6B.

Model: InternVideo2-Stage2_6B (vision-only, 224p, 4-frame segments)
Output: data/features/v1_video/{id}.npy -- shape [num_segments, 768]

Requirements:
  - InternVideo2 model code (see https://github.com/OpenGVLab/InternVideo)
  - Checkpoint: InternVideo2_S2_6B_vision.pt
  - pip install av torchvision

Usage:
  python feature_extraction/extract_visual.py --model-dir /path/to/internvideo2-6b
"""

import argparse
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

VIDEO_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/features/v1_video")

TARGET_FPS = 10
FRAMES_PER_SEGMENT = 4
RESIZE = 224

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

VISION_CONFIG_6B = {
    "embed_dim": 3200, "depth": 48, "num_heads": 25, "mlp_ratio": 4,
    "clip_embed_dim": 768, "attn_pool_num_heads": 16, "qkv_bias": False,
    "drop_path_rate": 0.25, "init_values": 0.00001, "qk_normalization": True,
    "use_flash_attn": False, "use_fused_rmsnorm": False, "use_fused_mlp": False,
    "fused_mlp_heuristic": 1, "layerscale_no_force_fp32": False,
    "num_frames": 4, "tubelet_size": 1, "sep_pos_embed": False,
    "sep_image_video_pos_embed": False, "use_checkpoint": False,
    "checkpoint_num": 0, "clip_teacher_embed_dim": 3200,
    "clip_teacher_final_dim": 768, "clip_norm_type": "l2",
    "clip_return_layer": 6, "clip_student_return_interval": 1,
}


def _gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
        )
        return len(out.strip().splitlines())
    except Exception:
        return 0


class _VisionEncoder(nn.Module):
    def __init__(self, vision_encoder: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[1]
        use_image = T == 1
        x = frames.permute(0, 2, 1, 3, 4)
        result = self.vision_encoder(x, None, use_image)
        return result[1]


def build_model(model_dir: Path, num_gpus: int) -> nn.Module:
    sys.path.insert(0, str(model_dir.resolve()))
    from modeling_internvideo2 import PretrainInternVideo2

    checkpoint_path = model_dir / "InternVideo2_S2_6B_vision.pt"
    ve = PretrainInternVideo2(in_chans=3, img_size=224, patch_size=14, **VISION_CONFIG_6B)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    sd = ckpt.get("module", ckpt) if isinstance(ckpt, dict) else ckpt

    ve_state = {
        k[len("vision_encoder."):].replace(".gamma", ".weight"): v
        for k, v in sd.items() if k.startswith("vision_encoder.")
    }
    if not ve_state:
        ve_state = {k.replace(".gamma", ".weight"): v for k, v in sd.items()}
    ve.load_state_dict(ve_state, strict=True)
    del ckpt, sd

    model = _VisionEncoder(ve).half().eval()
    if num_gpus > 1:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    else:
        model = model.cuda()
    return model


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESIZE),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def _decode_frames(video_path: Path) -> list[np.ndarray]:
    frames = []
    try:
        container = av.open(str(video_path))
    except Exception:
        return frames
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate or stream.guessed_rate or 30)
    if src_fps <= 0:
        src_fps = 30.0
    step = max(1, round(src_fps / TARGET_FPS))
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx % step == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_idx += 1
    container.close()
    return frames


def _segment_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    segments = []
    for i in range(0, len(frames), FRAMES_PER_SEGMENT):
        seg = frames[i : i + FRAMES_PER_SEGMENT]
        while len(seg) < FRAMES_PER_SEGMENT:
            seg.append(seg[-1])
        segments.append(np.stack(seg))
    return segments


def _preprocess_segments(segments: list[np.ndarray], transform) -> torch.Tensor:
    batch = []
    for seg in segments:
        t = torch.from_numpy(seg).permute(0, 3, 1, 2).float() / 255.0
        processed = torch.stack([transform(t[i]) for i in range(t.shape[0])])
        batch.append(processed)
    return torch.stack(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to InternVideo2-6B model directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()

    num_gpus = args.gpus or _gpu_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    total_batch = args.batch_size * num_gpus

    video_ids = sorted(p.stem for p in VIDEO_DIR.glob("*.mp4"))
    existing = {p.stem for p in OUTPUT_DIR.glob("*.npy")} if OUTPUT_DIR.exists() else set()
    remaining = [vid for vid in video_ids if vid not in existing]

    print(f"Total: {len(video_ids)}, Done: {len(existing)}, Remaining: {len(remaining)}")
    if not remaining:
        print("All videos already processed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(Path(args.model_dir), num_gpus)
    transform = _build_transform()

    for vid in tqdm(remaining, desc="Extracting"):
        video_path = VIDEO_DIR / f"{vid}.mp4"
        if not video_path.exists():
            continue
        try:
            frames = _decode_frames(video_path)
            if not frames:
                continue
            segments = _segment_frames(frames)
            input_tensor = _preprocess_segments(segments, transform).cuda().half()
            all_features = []
            with torch.no_grad():
                for i in range(0, input_tensor.shape[0], total_batch):
                    batch = input_tensor[i : i + total_batch]
                    feat = model(batch)
                    all_features.append(feat.cpu().float().numpy())
            features = np.concatenate(all_features, axis=0)
            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                np.save(OUTPUT_DIR / f"{vid}.npy", features)
        except Exception as e:
            tqdm.write(f"Error {vid}: {e}")


if __name__ == "__main__":
    main()
