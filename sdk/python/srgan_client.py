"""
srgan_client.py — Python client SDK for the SRGAN-Rust API.

Usage
-----
    from srgan_client import SrganClient

    client = SrganClient(base_url="https://api.srgan.example.com/api/v1",
                         api_key="sk-...")

    # Synchronous upscale → PIL Image
    image = client.upscale("photo.jpg", scale=4, model="natural")
    image.save("photo_4x.png")

    # Async workflow
    job_id = client.upscale_async("photo.jpg", scale=4, model="anime")
    status = client.get_job(job_id)
    client.download_result(job_id, "photo_4x_anime.png")

Requirements
------------
    pip install requests pillow
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Optional dependency: requests ────────────────────────────────────────────
try:
    import requests
    from requests import Response, Session
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "srgan_client requires 'requests'. Install it with: pip install requests"
    ) from exc

# ── Optional dependency: Pillow ──────────────────────────────────────────────
try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "srgan_client requires 'Pillow'. Install it with: pip install pillow"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SrganError(Exception):
    """Base class for all SRGAN client errors."""


class SrganAuthError(SrganError):
    """Raised on 401/403 responses."""


class SrganRateLimitError(SrganError):
    """Raised on 429 responses. ``retry_after`` is the suggested wait (seconds)."""

    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SrganNotFoundError(SrganError):
    """Raised on 404 responses."""


class SrganValidationError(SrganError):
    """Raised on 422 responses."""


class SrganServerError(SrganError):
    """Raised on 5xx responses."""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (plain dicts kept for zero-dependency friendliness)
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus:
    """Represents the current state of an async upscaling job."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def job_id(self) -> str:
        return self._data["job_id"]

    @property
    def status(self) -> str:
        """One of: queued, processing, completed, failed, cancelled."""
        return self._data["status"]

    @property
    def progress_pct(self) -> int:
        return self._data.get("progress_pct", 0)

    @property
    def result_url(self) -> Optional[str]:
        return self._data.get("result_url")

    @property
    def error(self) -> Optional[str]:
        return self._data.get("error")

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def __repr__(self) -> str:
        return (
            f"JobStatus(job_id={self.job_id!r}, status={self.status!r}, "
            f"progress={self.progress_pct}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main client
# ─────────────────────────────────────────────────────────────────────────────

class SrganClient:
    """
    Python client for the SRGAN-Rust REST API.

    Parameters
    ----------
    base_url:
        API base URL, e.g. ``"https://api.srgan.example.com/api/v1"``.
    api_key:
        API key sent as the ``X-API-Key`` header.
    jwt_token:
        Bearer JWT.  Supply either ``api_key`` or ``jwt_token``, not both.
    timeout:
        Default request timeout in seconds.
    max_retries:
        Number of automatic retries on rate-limit (429) responses.
    retry_backoff:
        Initial back-off between retries in seconds (doubles on each retry).
    """

    DEFAULT_TIMEOUT = 120
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_BACKOFF = 1.0

    def __init__(
        self,
        base_url: str = "http://localhost:8080/api/v1",
        *,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self._session = Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key
        elif jwt_token:
            self._session.headers["Authorization"] = f"Bearer {jwt_token}"
        self._session.headers["Accept"] = "application/json"

    # ── Public API ────────────────────────────────────────────────────────

    def upscale(
        self,
        image_path: str | Path,
        *,
        scale: int = 4,
        model: str = "natural",
        output_format: str = "png",
        quality: int = 85,
    ) -> Image.Image:
        """
        Upscale *image_path* synchronously and return a PIL ``Image``.

        Parameters
        ----------
        image_path:
            Path to the input image file.
        scale:
            Upscaling factor — 2, 4, or 8.
        model:
            Model name: ``"natural"``, ``"anime"``, or ``"bilinear"``.
        output_format:
            Output encoding: ``"png"``, ``"jpeg"``, or ``"webp"``.
        quality:
            JPEG/WebP quality (1–100, ignored for PNG).

        Returns
        -------
        PIL.Image.Image
            The upscaled image.
        """
        encoded = _encode_image(image_path)
        payload = {
            "image_base64": encoded,
            "scale_factor": scale,
            "model": model,
            "output_format": output_format,
            "quality": quality,
        }
        data = self._post("/upscale", payload)
        image_bytes = base64.b64decode(data["image_base64"])
        return Image.open(io.BytesIO(image_bytes))

    def upscale_async(
        self,
        image_path: str | Path,
        *,
        scale: int = 4,
        model: str = "natural",
        output_format: str = "png",
        quality: int = 85,
        priority: str = "normal",
        callback_url: Optional[str] = None,
    ) -> str:
        """
        Submit *image_path* for asynchronous upscaling.

        Returns
        -------
        str
            The ``job_id`` UUID string.  Use :meth:`get_job` to poll status.
        """
        encoded = _encode_image(image_path)
        payload: dict = {
            "image_base64": encoded,
            "scale_factor": scale,
            "model": model,
            "output_format": output_format,
            "quality": quality,
            "priority": priority,
        }
        if callback_url:
            payload["callback_url"] = callback_url

        data = self._post("/upscale/async", payload)
        return data["job_id"]

    def get_job(self, job_id: str) -> JobStatus:
        """Return the current :class:`JobStatus` for *job_id*."""
        data = self._get(f"/jobs/{job_id}")
        return JobStatus(data)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or in-progress job."""
        self._delete(f"/jobs/{job_id}")

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[JobStatus]:
        """List jobs for the authenticated client."""
        params: dict = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = self._get("/jobs", params=params)
        return [JobStatus(j) for j in data.get("jobs", [])]

    def download_result(
        self,
        job_id: str,
        output_path: str | Path,
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[JobStatus], None]] = None,
    ) -> Path:
        """
        Wait for *job_id* to complete, then download the result image.

        Parameters
        ----------
        job_id:
            The async job to wait for.
        output_path:
            Where to save the upscaled image.
        poll_interval:
            Seconds between status polls.
        timeout:
            Maximum seconds to wait before raising ``TimeoutError``.
        progress_callback:
            Called after each poll with the current :class:`JobStatus`.

        Returns
        -------
        pathlib.Path
            Resolved path to the saved image.
        """
        output_path = Path(output_path)
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            status = self.get_job(job_id)
            if progress_callback:
                progress_callback(status)

            if status.status == "completed":
                break
            if status.status == "failed":
                raise SrganError(f"Job {job_id} failed: {status.error}")
            if status.status == "cancelled":
                raise SrganError(f"Job {job_id} was cancelled")

            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for job {job_id}")

            time.sleep(poll_interval)

        # Download the result as binary
        resp = self._raw_get(f"/jobs/{job_id}/result")
        output_path.write_bytes(resp.content)
        return output_path.resolve()

    def list_models(self) -> list[dict]:
        """Return a list of available model descriptors."""
        data = self._get("/models")
        return data.get("models", [])

    def health(self) -> dict:
        """Return the service health status."""
        return self._get("/health")

    # ── Low-level helpers ─────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        return self._request_with_retry("POST", path, json=payload)

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request_with_retry("GET", path, params=params)

    def _delete(self, path: str) -> None:
        self._request_with_retry("DELETE", path, expect_json=False)

    def _raw_get(self, path: str) -> Response:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, timeout=self.timeout)
        _raise_for_status(resp)
        return resp

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        expect_json: bool = True,
        **kwargs,
    ) -> dict:
        url = f"{self.base_url}{path}"
        backoff = self.retry_backoff

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
            except requests.exceptions.ConnectionError as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "Connection error (attempt %d/%d), retrying in %.1fs…",
                        attempt + 1, self.max_retries + 1, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise SrganError(f"Connection failed: {exc}") from exc

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff))
                if attempt < self.max_retries:
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %ds…",
                        attempt + 1, self.max_retries + 1, retry_after,
                    )
                    time.sleep(retry_after)
                    backoff *= 2
                    continue
                raise SrganRateLimitError(
                    "Rate limit exceeded", retry_after=retry_after
                )

            _raise_for_status(resp)

            if not expect_json:
                return {}
            return resp.json()

        # Should not reach here
        raise SrganError("Exhausted retries")  # pragma: no cover


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(image_path: str | Path) -> str:
    """Read *image_path* and return a base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode()


def _raise_for_status(resp: Response) -> None:
    """Raise an appropriate :class:`SrganError` subclass for 4xx/5xx responses."""
    if resp.status_code < 400:
        return

    try:
        body = resp.json()
        message = body.get("error", resp.text)
        code = body.get("code", "")
    except Exception:
        message = resp.text
        code = ""

    if resp.status_code == 401:
        raise SrganAuthError(f"Unauthorized: {message}")
    if resp.status_code == 403:
        raise SrganAuthError(f"Forbidden: {message}")
    if resp.status_code == 404:
        raise SrganNotFoundError(message)
    if resp.status_code == 422:
        raise SrganValidationError(message)
    if resp.status_code >= 500:
        raise SrganServerError(f"Server error [{code}]: {message}")

    raise SrganError(f"HTTP {resp.status_code}: {message}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI convenience (python -m srgan_client)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="SRGAN-Rust Python SDK — quick CLI wrapper"
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--url", default="http://localhost:8080/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("SRGAN_API_KEY"))
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--model", default="natural",
                        choices=["natural", "anime", "bilinear"])
    parser.add_argument("--format", dest="fmt", default="png",
                        choices=["png", "jpeg", "webp"])
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--async-mode", action="store_true",
                        help="Submit as async job and poll until done")
    args = parser.parse_args()

    client = SrganClient(base_url=args.url, api_key=args.api_key)

    if args.async_mode:
        job_id = client.upscale_async(
            args.image,
            scale=args.scale,
            model=args.model,
            output_format=args.fmt,
            quality=args.quality,
        )
        print(f"Job submitted: {job_id}")
        result = client.download_result(
            job_id,
            args.output,
            progress_callback=lambda s: print(
                f"  {s.status} ({s.progress_pct}%)", end="\r", flush=True
            ),
        )
        print(f"\nSaved to {result}")
    else:
        img = client.upscale(
            args.image,
            scale=args.scale,
            model=args.model,
            output_format=args.fmt,
            quality=args.quality,
        )
        img.save(args.output)
        print(f"Saved {img.width}×{img.height} → {args.output}")

    sys.exit(0)
