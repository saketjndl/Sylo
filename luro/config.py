"""Luro SDK configuration.

Supports both programmatic configuration via luro.init() and
environment variable fallbacks. Environment variables are prefixed
with LURO_ and use uppercase naming.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from luro.exceptions import LuroConfigError

logger = logging.getLogger("luro")

# Valid values
VALID_ENVIRONMENTS = ("development", "staging", "production")
VALID_STORAGE_BACKENDS = ("local", "redis", "cloud")


class LuroConfig(BaseModel):
    """SDK configuration.

    All fields can be set programmatically via luro.init() or through
    environment variables. Programmatic values take precedence.

    Environment variables:
        LURO_API_KEY: API key for Luro Cloud
        LURO_PROJECT: Project name (required)
        LURO_ENVIRONMENT: development | staging | production
        LURO_STORAGE: local | redis | cloud
        LURO_REDIS_URL: Redis connection URL
        LURO_CLOUD_API_URL: Luro Cloud API base URL
    """

    project: str
    api_key: str | None = None
    environment: Literal["development", "staging", "production"] = "development"
    storage: Literal["local", "redis", "cloud"] = "local"
    redis_url: str = "redis://localhost:6379"
    cloud_api_url: str = "https://api.luro.dev"
    notifications: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def load_env_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Fall back to environment variables for any unset fields."""
        env_mapping = {
            "project": "LURO_PROJECT",
            "api_key": "LURO_API_KEY",
            "environment": "LURO_ENVIRONMENT",
            "storage": "LURO_STORAGE",
            "redis_url": "LURO_REDIS_URL",
            "cloud_api_url": "LURO_CLOUD_API_URL",
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


# ── Global config singleton ──────────────────────────────────────

_config: LuroConfig | None = None


def get_config() -> LuroConfig:
    """Get the current global configuration.

    Raises:
        LuroConfigError: If luro.init() has not been called yet.
    """
    if _config is None:
        raise LuroConfigError(
            "Luro is not initialized. Call luro.init(project='my-project') first."
        )
    return _config


def set_config(config: LuroConfig) -> None:
    """Set the global configuration. Called by luro.init()."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration. Primarily used in tests."""
    global _config
    _config = None
