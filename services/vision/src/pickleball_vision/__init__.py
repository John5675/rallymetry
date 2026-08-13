"""Pickleball Vision service package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pickleball-vision")
except PackageNotFoundError:  # pragma: no cover - supports direct source inspection
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
