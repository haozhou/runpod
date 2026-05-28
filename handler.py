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


# RunPod model caching 把模型放在 HF cache 标准结构下
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
# 这里填的字符串必须和 endpoint 设置里 Model 字段填的 HF repo id 完全一致
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_snapshot_path(model_id: str) -> str:
    """把 'org/model' 解析成 /runpod-volume/.../snapshots/<hash>/ 的真实路径."""
    safe_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(HF_CACHE_ROOT, safe_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(
            f"未找到缓存目录: {snapshots_dir}\n"
            f"请确认 endpoint 的 Model 字段填的是 '{model_id}',且 worker 已完成首次缓存。"
        )
    snapshots = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snapshots:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshots_dir}")
    # 有多个 snapshot 时取最新的一个
    snapshots.sort(
        key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)),
        reverse=True,
    )
    return os.path.join(snapshots_dir, snapshots[0])


MODEL_PATH = resolve_snapshot_path(MODEL_ID)
print(f"Loading model {MODEL_ID} from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE)
print("Model loaded successfully.")


# Z-Image-Turbo 官方推荐区间,可通过环境变量覆盖默认值
DEFAULT_NUM_STEPS = int(os.getenv("DEFAULT_NUM_STEPS", "30"))               # 推荐 28-50
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "4.0"))  # 推荐 3.0-5.0
print(f"Default num_inference_steps: {DEFAULT_NUM_STEPS}")
print(f"Default guidance_scale: {DEFAULT_GUIDANCE_SCALE}")


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
