"""
srgan-sdk — Python SDK for the SRGAN-Rust image upscaling API.

Quickstart
----------
    from srgan_sdk import SrganClient

    client = SrganClient("http://localhost:8080", api_key="sk-...")

    # Synchronous upscale
    result = client.upscale("photo.jpg", model="esrgan", scale=4)
    with open("photo_4x.png", "wb") as f:
        import base64
        f.write(base64.b64decode(result.image))

    # Async workflow
    job_id = client.upscale_async("photo.jpg")
    status = client.status(job_id)
"""

from srgan_sdk.client import SrganClient
from srgan_sdk.models import (
    JobStatus,
    ModelInfo,
    UpscaleResult,
)
from srgan_sdk.exceptions import (
    SrganError,
    SrganAuthError,
    SrganRateLimitError,
    SrganValidationError,
    SrganNotFoundError,
    SrganServerError,
)

__all__ = [
    "SrganClient",
    # result types
    "UpscaleResult",
    "JobStatus",
    "ModelInfo",
    # exceptions
    "SrganError",
    "SrganAuthError",
    "SrganRateLimitError",
    "SrganValidationError",
    "SrganNotFoundError",
    "SrganServerError",
]

__version__ = "0.1.0"
