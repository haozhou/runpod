import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io
import time
import base64
import traceback
import torch
import runpod
from diffusers import ZImagePipeline


HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
# 旧的 network volume fallback 路径(如果 model caching 不生效就用这个)
FALLBACK_MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def debug_dump_volume():
    """启动时打印 /runpod-volume 的目录结构,方便排查 cache 状态."""
    print("=" * 60)
    print("DEBUG: /runpod-volume contents")
    print("=" * 60)
    for root, dirs, files in os.walk("/runpod-volume"):
        depth = root.replace("/runpod-volume", "").count(os.sep)
        if depth > 4:  # 限制深度避免日志爆炸
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or '/runpod-volume'}/")
        sub_indent = "  " * (depth + 1)
        for f in files[:5]:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... ({len(files) - 5} more files)")
    print("=" * 60)


def resolve_snapshot_path(model_id: str):
    safe_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(HF_CACHE_ROOT, safe_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    snapshots = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snapshots:
        return None
    snapshots.sort(
        key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)),
        reverse=True,
    )
    return os.path.join(snapshots_dir, snapshots[0])


def pick_model_path() -> str:
    cached = resolve_snapshot_path(MODEL_ID)
    if cached:
        print(f"[OK] Using RunPod model cache: {cached}")
        return cached
    print(f"[WARN] No cached snapshot for {MODEL_ID} at {HF_CACHE_ROOT}")
    if os.path.isdir(FALLBACK_MODEL_DIR):
        print(f"[OK] Falling back to: {FALLBACK_MODEL_DIR}")
        return FALLBACK_MODEL_DIR
    raise FileNotFoundError(
        f"Neither RunPod model cache nor fallback dir is available.\n"
        f"  Tried cache: {HF_CACHE_ROOT}/models--{MODEL_ID.replace('/', '--')}/snapshots\n"
        f"  Tried fallback: {FALLBACK_MODEL_DIR}"
    )


debug_dump_volume()
MODEL_PATH = pick_model_path()
print(f"Loading model from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE)
print("Model loaded successfully.")

# ... handler 函数和之前一样,这里省略
