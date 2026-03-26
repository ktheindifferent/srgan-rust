"""Exceptions raised by the SRGAN SDK."""

from __future__ import annotations


class SrganError(Exception):
    """Base class for all SRGAN SDK errors."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class SrganAuthError(SrganError):
    """Raised on 401/403 responses (invalid or missing API key)."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401)


class SrganRateLimitError(SrganError):
    """Raised on 429 responses.  ``retry_after`` is the suggested wait in seconds."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class SrganValidationError(SrganError):
    """Raised on 422 responses (bad request body)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class SrganNotFoundError(SrganError):
    """Raised on 404 responses."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class SrganServerError(SrganError):
    """Raised on 5xx responses."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)
