"""Data models returned by the SRGAN API."""

from __future__ import annotations

from typing import List, Optional


class UpscaleResult:
    """
    Result of a completed upscale operation.

    Attributes
    ----------
    image:
        Base64-encoded upscaled image bytes.
    content_type:
        MIME type of the output image (e.g. ``"image/png"``).
    width:
        Original width in pixels.
    height:
        Original height in pixels.
    inference_ms:
        Server-side inference duration in milliseconds.
    """

    __slots__ = ("image", "content_type", "width", "height", "inference_ms")

    def __init__(self, data: dict) -> None:
        self.image: str = data["image"]
        self.content_type: str = data.get("content_type", "image/png")
        self.width: int = data.get("width", 0)
        self.height: int = data.get("height", 0)
        self.inference_ms: int = data.get("inference_ms", 0)

    def save(self, path: str) -> None:
        """Decode the base64 image and write it to *path*."""
        import base64

        with open(path, "wb") as fh:
            fh.write(base64.b64decode(self.image))

    def __repr__(self) -> str:
        return (
            f"UpscaleResult(content_type={self.content_type!r}, "
            f"size={self.width}x{self.height}, inference_ms={self.inference_ms})"
        )


class JobStatus:
    """
    Status of an asynchronous upscale job.

    Attributes
    ----------
    job_id:
        Unique job identifier.
    status:
        One of ``"pending"``, ``"processing"``, ``"completed"``, ``"failed"``,
        ``"timed_out"``.
    created_at:
        ISO-8601 creation timestamp.
    completed_at:
        ISO-8601 completion timestamp, or ``None`` if not yet finished.
    result:
        :class:`UpscaleResult` populated when ``status == "completed"``.
    error:
        Error message populated when ``status == "failed"``.
    """

    __slots__ = ("job_id", "status", "created_at", "completed_at", "result", "error")

    def __init__(self, data: dict) -> None:
        self.job_id: str = data["job_id"]
        self.status: str = data["status"]
        self.created_at: str = data.get("created_at", "")
        self.completed_at: Optional[str] = data.get("completed_at")
        self.result: Optional[UpscaleResult] = (
            UpscaleResult(data["result"]) if data.get("result") else None
        )
        self.error: Optional[str] = data.get("error")

    @property
    def is_terminal(self) -> bool:
        """``True`` if the job has reached a terminal state."""
        return self.status in ("completed", "failed", "timed_out")

    def __repr__(self) -> str:
        return f"JobStatus(job_id={self.job_id!r}, status={self.status!r})"


class ModelInfo:
    """
    Metadata about a model available on the server.

    Attributes
    ----------
    label:
        Short identifier used in API requests (e.g. ``"esrgan"``).
    description:
        Human-readable description.
    scale_factor:
        Native upscaling factor for this model.
    """

    __slots__ = ("label", "description", "scale_factor")

    def __init__(self, data: dict) -> None:
        self.label: str = data["label"]
        self.description: str = data.get("description", "")
        self.scale_factor: int = data.get("scale_factor", 4)

    def __repr__(self) -> str:
        return (
            f"ModelInfo(label={self.label!r}, scale_factor={self.scale_factor}, "
            f"description={self.description!r})"
        )
