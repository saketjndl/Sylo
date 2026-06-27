"""Storage backend factory and package exports.

Use get_storage() to obtain the correct backend based on the
current SDK configuration.
"""

from __future__ import annotations

from sylo.config import SyloConfig
from sylo.storage.base import SyloStorage, LuroStorage
from sylo.storage.cloud_store import CloudStorage
from sylo.storage.local_store import LocalStorage
from sylo.storage.redis_store import RedisStorage

__all__ = [
    "SyloStorage",
    "LuroStorage",
    "LocalStorage",
    "RedisStorage",
    "CloudStorage",
    "get_storage",
]


def get_storage(config: SyloConfig) -> SyloStorage:
    """Create the appropriate storage backend from config.

    Args:
        config: The current SDK configuration.

    Returns:
        A storage backend instance ready for use.

    Raises:
        ValueError: If the configured storage backend is not recognized.
    """
    if config.storage == "local":
        return LocalStorage()
    elif config.storage == "redis":
        return RedisStorage(redis_url=config.redis_url)
    elif config.storage == "cloud":
        if config.api_key is None:
            raise ValueError(
                "Cloud storage requires an API key. "
                "Set api_key in sylo.init() or SYLO_API_KEY env var."
            )
        return CloudStorage(
            api_key=config.api_key,
            base_url=config.cloud_api_url,
        )
    else:
        raise ValueError(f"Unknown storage backend: {config.storage!r}")
