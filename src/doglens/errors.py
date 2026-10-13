"""Domain exceptions carrying a machine-readable code and a Russian user-facing message.

The UI shows ``user_message`` only; tracebacks and technical details go to logs, which must
never contain image content, EXIF/GPS or the user's original file name.
"""

from __future__ import annotations


class DogLensError(Exception):
    """Base class for expected, user-presentable failures."""

    def __init__(self, code: str, user_message: str, detail: str | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.user_message = user_message
        self.detail = detail


class ImageValidationError(DogLensError):
    """The uploaded bytes are not an acceptable single-frame JPEG/PNG/WebP image."""


class RoiError(DogLensError):
    """The requested region of interest is invalid."""


class ConfigError(DogLensError):
    """A configuration file is missing or invalid."""

    def __init__(self, detail: str) -> None:
        super().__init__("invalid_config", "Некорректная конфигурация.", detail)
