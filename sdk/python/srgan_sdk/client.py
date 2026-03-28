"""
SrganClient — typed Python client for the SRGAN-Rust REST API.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
from typing_extensions import Literal

from srgan_sdk.exceptions import (
    SrganAuthError,
    SrganError,
    SrganNotFoundError,
    SrganRateLimitError,
    SrganServerError,
    SrganValidationError,
)
from srgan_sdk.models import JobStatus, ModelInfo, UpscaleResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 1.0

ModelLabel = Literal[
    "esrgan", "esrgan-anime", "esrgan-x2", "natural", "anime", "bilinear",
    "waifu2x", "waifu2x-anime", "waifu2x-photo",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SrganClient:
    """
    Synchronous client for the SRGAN-Rust image upscaling API.

    Parameters
    ----------
    base_url:
        Base URL of the SRGAN server, e.g. ``"http://localhost:8080"``.
        Trailing slash is stripped automatically.
    api_key:
        Optional API key sent as the ``X-API-Key`` request header.
    timeout:
        Per-request timeout in seconds (default: 120).
    max_retries:
        Number of automatic retries on transient network errors and 429
        responses (default: 3).
    retry_backoff:
        Initial back-off between retries in seconds; doubles on each retry
        (default: 1.0).

    Examples
    --------
    >>> client = SrganClient("http://localhost:8080", api_key="sk-...")
    >>> result = client.upscale("photo.jpg")
    >>> result.save("photo_4x.png")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            headers["X-API-Key"] = api_key

        self._http = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def upscale(
        self,
        image: Union[str, Path, bytes],
        *,
        model: str = "esrgan",
        scale: int = 4,
        waifu2x_noise_level: Optional[int] = None,
        waifu2x_scale: Optional[int] = None,
        waifu2x_style: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> UpscaleResult:
        """
        Upscale an image synchronously.

        For large images the server may queue the job and return 202.  In that
        case this method automatically polls until the job completes.

        Parameters
        ----------
        image:
            A file path (``str`` or :class:`pathlib.Path`) or raw image
            bytes.  File paths are read and base64-encoded automatically.
        model:
            Model identifier to use (default: ``"esrgan"``).  For waifu2x,
            pass ``"waifu2x"`` or a variant label like
            ``"waifu2x-noise2-scale4"``.
        scale:
            Upscaling factor — typically 2 or 4 (default: ``4``).
        waifu2x_noise_level:
            Waifu2x noise reduction level 0–3 (default: 1).  Only used
            when *model* starts with ``"waifu2x"``.
        waifu2x_scale:
            Waifu2x scale factor: 1 (denoise-only), 2, 3, or 4.  Only
            used when *model* starts with ``"waifu2x"``.
        waifu2x_style:
            Waifu2x content style: ``"anime"`` (default), ``"photo"``, or
            ``"artwork"``.  Only used when *model* starts with
            ``"waifu2x"``.
        webhook_url:
            Optional URL the server will POST the result to when the job
            finishes.

        Returns
        -------
        UpscaleResult
            The upscaled image wrapped in an :class:`~srgan_sdk.models.UpscaleResult`.
        """
        payload: dict = {
            "image": _to_base64(image),
            "model": model,
            "scale": scale,
        }
        if waifu2x_noise_level is not None:
            payload["waifu2x_noise_level"] = waifu2x_noise_level
        if waifu2x_scale is not None:
            payload["waifu2x_scale"] = waifu2x_scale
        if waifu2x_style is not None:
            payload["waifu2x_style"] = waifu2x_style
        if webhook_url is not None:
            payload["webhook_url"] = webhook_url

        data = self._request("POST", "/api/v1/upscale", json=payload)

        # Server may return 202 with a job_id for large images.
        if "job_id" in data:
            return self._wait_for_job(data["job_id"])

        return UpscaleResult(data)

    def upscale_async(
        self,
        image: Union[str, Path, bytes],
        *,
        model: str = "esrgan",
        scale: int = 4,
        webhook_url: Optional[str] = None,
    ) -> str:
        """
        Submit an upscale job asynchronously (fire-and-forget).

        Parameters
        ----------
        image:
            A file path or raw image bytes.
        model:
            Model identifier to use (default: ``"esrgan"``).
        scale:
            Upscaling factor (default: ``4``).
        webhook_url:
            Optional webhook URL the server will POST the result to.

        Returns
        -------
        str
            The ``job_id`` string.  Use :meth:`status` to poll for completion.
        """
        payload: dict = {
            "image": _to_base64(image),
            "model": model,
            "scale": scale,
        }
        if webhook_url is not None:
            payload["webhook_url"] = webhook_url

        data = self._request("POST", "/api/v1/upscale/async", json=payload)
        return str(data["job_id"])

    def status(self, job_id: str) -> JobStatus:
        """
        Return the current :class:`~srgan_sdk.models.JobStatus` for *job_id*.

        Parameters
        ----------
        job_id:
            The job identifier returned by :meth:`upscale_async`.

        Returns
        -------
        JobStatus
        """
        data = self._request("GET", f"/api/v1/job/{_quote(job_id)}")
        return JobStatus(data)

    def health(self) -> dict:
        """
        Return the server health status.

        Returns
        -------
        dict
            A dictionary with at least ``status``, ``version``,
            ``uptime_seconds``, ``model_loaded``, and ``images_processed``.
        """
        return self._request("GET", "/api/health")

    def models(self) -> List[ModelInfo]:
        """
        Return a list of models available on the server.

        Returns
        -------
        list[ModelInfo]
        """
        data = self._request("GET", "/api/models")
        # Accept either a bare list or ``{"models": [...]}``
        items = data if isinstance(data, list) else data.get("models", [])
        return [ModelInfo(m) for m in items]

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "SrganClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Private helpers ───────────────────────────────────────────────────

    def _wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> UpscaleResult:
        """Poll *job_id* until it reaches a terminal state and return the result."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            js = self.status(job_id)
            if js.status == "completed":
                if js.result is None:
                    raise SrganError(f"Job {job_id} completed but result is missing")
                return js.result
            if js.status == "failed":
                raise SrganError(f"Job {job_id} failed: {js.error or 'unknown error'}")
            if js.status == "timed_out":
                raise SrganError(f"Job {job_id} timed out on the server")
            time.sleep(poll_interval)

        raise SrganError(f"Timed out waiting for job {job_id} after {timeout:.0f}s")

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Execute *method* on *path* with automatic retry/backoff."""
        backoff = self._retry_backoff
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Transport error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise SrganError(f"Request failed after {self._max_retries + 1} attempts: {exc}") from exc

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", str(int(backoff))))
                if attempt < self._max_retries:
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        self._max_retries + 1,
                        retry_after,
                    )
                    time.sleep(retry_after)
                    backoff *= 2
                    continue
                raise SrganRateLimitError("Rate limit exceeded", retry_after=retry_after)

            _raise_for_status(response)

            # 204 No Content
            if response.status_code == 204:
                return {}

            return response.json()

        raise SrganError(f"Request failed: {last_exc}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_base64(image: Union[str, Path, bytes]) -> str:
    """Return a base64 string from a file path or raw bytes."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        raw = path.read_bytes()
    else:
        raw = image
    return base64.b64encode(raw).decode()


def _quote(value: str) -> str:
    """Percent-encode *value* for safe use in URL path segments."""
    from urllib.parse import quote

    return quote(str(value), safe="")


def _raise_for_status(response: httpx.Response) -> None:
    """Raise an appropriate :class:`SrganError` subclass for 4xx/5xx responses."""
    code = response.status_code
    if code < 400:
        return

    try:
        body = response.json()
        message: str = body.get("error") or body.get("message") or response.text
    except Exception:
        message = response.text or f"HTTP {code}"

    if code == 401 or code == 403:
        raise SrganAuthError(message)
    if code == 404:
        raise SrganNotFoundError(message)
    if code == 422:
        raise SrganValidationError(message)
    if code >= 500:
        raise SrganServerError(message, status_code=code)

    raise SrganError(f"HTTP {code}: {message}", status_code=code)
