"""
Basic usage examples for the srgan-sdk Python package.

Run with:
    pip install srgan-sdk
    python examples/basic.py
"""

import os
import sys
from pathlib import Path

from srgan_sdk import SrganClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("SRGAN_BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("SRGAN_API_KEY")  # None for local servers


# ---------------------------------------------------------------------------
# Example 1: Synchronous upscale from a file path
# ---------------------------------------------------------------------------

def example_upscale_file(input_path: str, output_path: str) -> None:
    """Upscale an image file and save the result."""
    with SrganClient(BASE_URL, api_key=API_KEY) as client:
        print(f"Upscaling {input_path!r} ...")
        result = client.upscale(input_path, model="esrgan", scale=4)
        result.save(output_path)
        print(
            f"Done — {result.width}x{result.height} → saved to {output_path!r} "
            f"({result.inference_ms} ms inference)"
        )


# ---------------------------------------------------------------------------
# Example 2: Synchronous upscale from raw bytes
# ---------------------------------------------------------------------------

def example_upscale_bytes(image_bytes: bytes, output_path: str) -> None:
    """Upscale raw image bytes and save the result."""
    with SrganClient(BASE_URL, api_key=API_KEY) as client:
        result = client.upscale(image_bytes, model="esrgan-anime", scale=4)
        result.save(output_path)
        print(f"Upscaled from bytes → {output_path!r}")


# ---------------------------------------------------------------------------
# Example 3: Async job submission + polling
# ---------------------------------------------------------------------------

def example_async_upscale(input_path: str, output_path: str) -> None:
    """Submit an async job and poll until completion."""
    import time

    with SrganClient(BASE_URL, api_key=API_KEY) as client:
        print(f"Submitting async job for {input_path!r} ...")
        job_id = client.upscale_async(input_path, model="esrgan", scale=4)
        print(f"Job ID: {job_id}")

        while True:
            js = client.status(job_id)
            print(f"  status: {js.status}")
            if js.is_terminal:
                break
            time.sleep(2.0)

        if js.status == "completed" and js.result:
            js.result.save(output_path)
            print(f"Result saved to {output_path!r}")
        else:
            print(f"Job failed: {js.error}", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Example 4: Webhook-triggered async job
# ---------------------------------------------------------------------------

def example_webhook(input_path: str, webhook: str) -> str:
    """Submit a job with a webhook URL; returns the job_id."""
    with SrganClient(BASE_URL, api_key=API_KEY) as client:
        job_id = client.upscale_async(
            input_path,
            model="esrgan",
            scale=4,
            webhook_url=webhook,
        )
        print(f"Job {job_id!r} submitted; result will be POSTed to {webhook!r}")
        return job_id


# ---------------------------------------------------------------------------
# Example 5: Service introspection
# ---------------------------------------------------------------------------

def example_service_info() -> None:
    """Print server health and available models."""
    with SrganClient(BASE_URL, api_key=API_KEY) as client:
        info = client.health()
        print("Health:", info)

        models = client.models()
        print(f"\nAvailable models ({len(models)}):")
        for m in models:
            print(f"  {m.label:<20} x{m.scale_factor}  {m.description}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test using the first argument as input image, if provided.
    if len(sys.argv) >= 2:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) >= 3 else "output_4x.png"
        example_upscale_file(inp, out)
    else:
        # Just show service info if no image is passed.
        example_service_info()
