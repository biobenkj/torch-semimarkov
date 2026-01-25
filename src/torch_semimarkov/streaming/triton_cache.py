"""Triton cache validation utilities.

This module provides utilities to detect when Triton kernel configuration
(num_warps, tile_c) has changed, helping users avoid stale cache issues.

While Triton's cache key includes num_warps, this sentinel file provides
an explicit warning when configuration changes, making debugging easier.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple


class TritonConfig(NamedTuple):
    """Configuration values that affect Triton kernel compilation."""

    num_warps: int
    tile_c: int = 16  # For backward kernel


def get_triton_cache_dir() -> Path:
    """Get Triton cache directory, respecting TRITON_CACHE_DIR env var."""
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".triton" / "cache"


def _config_hash(config: TritonConfig) -> str:
    """Generate a hash of the config for validation."""
    data = json.dumps(config._asdict(), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _sentinel_path() -> Path:
    """Path to the config sentinel file."""
    return get_triton_cache_dir() / ".torch_semimarkov_config"


def validate_triton_cache(config: TritonConfig, warn: bool = True) -> bool:
    """Check if Triton cache matches current config.

    Args:
        config: Current Triton config (num_warps, tile_c)
        warn: If True, print warning on mismatch

    Returns:
        True if cache is valid (or no sentinel exists), False on mismatch
    """
    sentinel = _sentinel_path()
    current_hash = _config_hash(config)

    if not sentinel.exists():
        # No sentinel - first run or cache was cleared
        # Write current config as sentinel
        try:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(
                json.dumps(
                    {
                        "config": config._asdict(),
                        "hash": current_hash,
                    }
                )
            )
        except OSError:
            pass  # Cache dir not writable, skip validation
        return True

    try:
        stored = json.loads(sentinel.read_text())
        stored_hash = stored.get("hash", "")

        if stored_hash != current_hash:
            if warn:
                stored_config = stored.get("config", {})
                import warnings

                warnings.warn(
                    f"Triton cache may be stale! "
                    f"Config changed from {stored_config} to {config._asdict()}. "
                    f"Consider running: rm -rf {get_triton_cache_dir()}",
                    UserWarning,
                    stacklevel=3,
                )
            return False

        return True

    except (json.JSONDecodeError, OSError):
        return True  # Can't read sentinel, assume OK


def update_cache_sentinel(config: TritonConfig) -> None:
    """Update sentinel after successful kernel compilation."""
    sentinel = _sentinel_path()
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            json.dumps(
                {
                    "config": config._asdict(),
                    "hash": _config_hash(config),
                }
            )
        )
    except OSError:
        pass


def clear_cache_sentinel() -> None:
    """Remove sentinel (call when user clears cache)."""
    try:
        _sentinel_path().unlink(missing_ok=True)
    except OSError:
        pass
