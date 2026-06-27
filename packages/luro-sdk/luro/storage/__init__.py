"""Storage backend factory and package exports.

Use get_storage() to obtain the correct backend based on the
current SDK configuration.
"""

from __future__ import annotations

from luro.config import LuroConfig
from luro.storage.base import LuroStorage
from luro.storage.cloud_store import CloudStorage
from luro.storage.local_store import LocalStorage
from luro.storage.redis_store import RedisStorage

__all__ = [
    "LuroStorage",
    "LocalStorage",
    "RedisStorage",
    "CloudStorage",
    "get_storage",
]


def get_storage(config: LuroConfig) -> LuroStorage:
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
                "Set api_key in luro.init() or LURO_API_KEY env var."
            )
        return CloudStorage(
            api_key=config.api_key,
            base_url=config.cloud_api_url,
        )
    else:
        raise ValueError(f"Unknown storage backend: {config.storage!r}")
