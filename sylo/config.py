"""Sylo SDK configuration.

Supports both programmatic configuration via sylo.init() and
environment variable fallbacks. Environment variables are prefixed
with SYLO_ and use uppercase naming.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sylo.exceptions import SyloConfigError

logger = logging.getLogger("sylo")

# Valid values
VALID_ENVIRONMENTS = ("development", "staging", "production")
VALID_STORAGE_BACKENDS = ("local", "redis", "cloud")


class SyloConfig(BaseModel):
    """SDK configuration.

    All fields can be set programmatically via sylo.init() or through
    environment variables. Programmatic values take precedence.

    Environment variables:
        SYLO_API_KEY: API key for Sylo Cloud
        SYLO_PROJECT: Project name (required)
        SYLO_ENVIRONMENT: development | staging | production
        SYLO_STORAGE: local | redis | cloud
        SYLO_REDIS_URL: Redis connection URL
        SYLO_CLOUD_API_URL: Sylo Cloud API base URL
    """

    project: str
    api_key: str | None = None
    environment: Literal["development", "staging", "production"] = "development"
    storage: Literal["local", "redis", "cloud"] = "local"
    redis_url: str = "redis://localhost:6379"
    cloud_api_url: str = "https://api.sylo.dev"
    notifications: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def load_env_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Fall back to environment variables for any unset fields."""
        env_mapping = {
            "project": "SYLO_PROJECT",
            "api_key": "SYLO_API_KEY",
            "environment": "SYLO_ENVIRONMENT",
            "storage": "SYLO_STORAGE",
            "redis_url": "SYLO_REDIS_URL",
            "cloud_api_url": "SYLO_CLOUD_API_URL",
        }
        for field_name, env_var in env_mapping.items():
            if field_name not in data or data[field_name] is None:
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    data[field_name] = env_value
        return data

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def has_cloud(self) -> bool:
        """Check if cloud sync is enabled (API key is set)."""
        return self.api_key is not None


# Backwards compatibility alias
LuroConfig = SyloConfig

# ── Global config singleton ──────────────────────────────────────

_config: SyloConfig | None = None


def get_config() -> SyloConfig:
    """Get the current global configuration.

    Raises:
        SyloConfigError: If sylo.init() has not been called yet.
    """
    if _config is None:
        raise SyloConfigError(
            "Sylo is not initialized. Call sylo.init(project='my-project') first."
        )
    return _config


def set_config(config: SyloConfig) -> None:
    """Set the global configuration. Called by sylo.init()."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration. Primarily used in tests."""
    global _config
    _config = None
