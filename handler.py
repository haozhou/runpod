import io
import os
import time
import base64
import traceback

import torch
import runpod
from diffusers import ZImagePipeline

MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_DIR}")
print(f"Using device: {DEVICE}")

pipe = ZImagePipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE)

print("Model loaded successfully.")

def handler(job):
    try:
        job_input = job.get("input", {})

        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        negative_prompt = job_input.get("negative_prompt", None)
        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        num_inference_steps = int(job_input.get("num_inference_steps", 9))
        seed = int(job_input.get("seed", 42))

        # 这个 Turbo 模型建议 guidance_scale 固定 0.0
        guidance_scale = 0.0

        # 如果你想固定只允许 1024x1024
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
            "traceback": traceback.format_exc()
        }

runpod.serverless.start({"handler": handler})
