"""This package contains general utilities to build fractal tasks."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-tasks-utils")
except PackageNotFoundError:
    __version__ = "unknown"
