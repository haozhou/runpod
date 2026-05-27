import os

# 必须在 import diffusers/transformers 之前设置,防止启动时联网回 HF
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
# 注意:用的是基础模型 Z-Image,不是 Turbo
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_snapshot_path(model_id: str):
    """
    在 HF_CACHE_ROOT 下查找 model_id 对应的最新 snapshot 路径。
    RunPod 会把目录名小写化,所以做大小写不敏感的匹配。
    """
    target = ("models--" + model_id.replace("/", "--")).lower()

    if not os.path.isdir(HF_CACHE_ROOT):
        return None

    matched_dir = None
    for d in os.listdir(HF_CACHE_ROOT):
        if d.lower() == target and os.path.isdir(os.path.join(HF_CACHE_ROOT, d)):
            matched_dir = d
            break
    if matched_dir is None:
        return None

    snapshots_dir = os.path.join(HF_CACHE_ROOT, matched_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snapshots:
        return None

    # 有多个 snapshot 时取最新一个
    snapshots.sort(
        key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)),
        reverse=True,
    )
    return os.path.join(snapshots_dir, snapshots[0])


MODEL_PATH = resolve_snapshot_path(MODEL_ID)
if MODEL_PATH is None:
    raise FileNotFoundError(
        f"未在 {HF_CACHE_ROOT} 下找到 {MODEL_ID} 的缓存。\n"
        f"请确认 endpoint 的 Model 字段填的是 '{MODEL_ID}',并已完成首次缓存。"
    )

print(f"Loading model {MODEL_ID} from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE)
print("Model loaded successfully.")


# Z-Image 基础模型(非 Turbo)的推荐参数
DEFAULT_NUM_STEPS = 30        # 推荐 28–50
DEFAULT_GUIDANCE_SCALE = 4.0  # 推荐 3.0–5.0


def handler(job):
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        negative_prompt = job_input.get("negative_prompt", None)
        num_inference_steps = int(job_input.get("num_inference_steps", DEFAULT_NUM_STEPS))
        guidance_scale = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        seed = int(job_input.get("seed", 42))

        # 锁定 1024x1024
        width = 1024
        height = 1024

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = round(time.time() - start, 2)

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": image_base64,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
