"""Tests for Luro SDK configuration."""

from __future__ import annotations

import os

import pytest

import luro
from luro.config import LuroConfig, get_config, reset_config, set_config
from luro.exceptions import LuroConfigError


class TestLuroConfig:
    """Tests for LuroConfig model validation."""

    def test_minimal_config(self):
        """Config with just project name should work."""
        config = LuroConfig(project="my-project")
        assert config.project == "my-project"
        assert config.api_key is None
        assert config.environment == "development"
        assert config.storage == "local"

    def test_full_config(self):
        """All fields should be settable."""
        config = LuroConfig(
            project="my-project",
            api_key="luro_test_key",
            environment="production",
            storage="redis",
            redis_url="redis://myhost:6380",
        )
        assert config.project == "my-project"
        assert config.api_key == "luro_test_key"
        assert config.environment == "production"
        assert config.storage == "redis"
        assert config.redis_url == "redis://myhost:6380"

    def test_invalid_environment_rejected(self):
        """Invalid environment values should raise validation error."""
        with pytest.raises(Exception):
            LuroConfig(project="test", environment="invalid")

    def test_invalid_storage_rejected(self):
        """Invalid storage values should raise validation error."""
        with pytest.raises(Exception):
            LuroConfig(project="test", storage="mongodb")

    def test_env_var_fallback(self):
        """Environment variables should fill in missing config values."""
        os.environ["LURO_PROJECT"] = "env-project"
        os.environ["LURO_API_KEY"] = "luro_env_key"
        os.environ["LURO_ENVIRONMENT"] = "staging"
        try:
            config = LuroConfig()  # type: ignore[call-arg]
            assert config.project == "env-project"
            assert config.api_key == "luro_env_key"
            assert config.environment == "staging"
        finally:
            del os.environ["LURO_PROJECT"]
            del os.environ["LURO_API_KEY"]
            del os.environ["LURO_ENVIRONMENT"]

    def test_programmatic_overrides_env(self):
        """Programmatic values should take precedence over env vars."""
        os.environ["LURO_PROJECT"] = "env-project"
        try:
            config = LuroConfig(project="programmatic-project")
            assert config.project == "programmatic-project"
        finally:
            del os.environ["LURO_PROJECT"]

    def test_is_development(self):
        """is_development property should reflect environment."""
        config = LuroConfig(project="test", environment="development")
        assert config.is_development is True
        assert config.is_production is False

    def test_is_production(self):
        """is_production property should reflect environment."""
        config = LuroConfig(project="test", environment="production")
        assert config.is_production is True
        assert config.is_development is False

    def test_has_cloud(self):
        """has_cloud should be True only when api_key is set."""
        config_no_key = LuroConfig(project="test")
        assert config_no_key.has_cloud is False

        config_with_key = LuroConfig(project="test", api_key="luro_xxx")
        assert config_with_key.has_cloud is True


class TestGlobalConfig:
    """Tests for global config singleton."""

    def test_get_config_before_init_raises(self):
        """Accessing config before init should raise LuroConfigError."""
        with pytest.raises(LuroConfigError, match="not initialized"):
            get_config()

    def test_set_and_get_config(self):
        """Setting config should make it retrievable."""
        config = LuroConfig(project="test")
        set_config(config)
        assert get_config().project == "test"

    def test_reset_config(self):
        """Resetting config should make get_config raise again."""
        set_config(LuroConfig(project="test"))
        reset_config()
        with pytest.raises(LuroConfigError):
            get_config()


class TestLuroInit:
    """Tests for the luro.init() function."""

    def test_basic_init(self):
        """luro.init() should set global config."""
        luro.init(project="my-project")
        config = get_config()
        assert config.project == "my-project"
        assert config.environment == "development"
        assert config.storage == "local"

    def test_init_with_all_options(self):
        """luro.init() should accept all config options."""
        luro.init(
            project="my-project",
            api_key="luro_key",
            environment="production",
            storage="local",
        )
        config = get_config()
        assert config.project == "my-project"
        assert config.api_key == "luro_key"
        assert config.environment == "production"

    def test_init_without_project_raises(self):
        """luro.init() without project should raise LuroConfigError."""
        with pytest.raises(LuroConfigError):
            luro.init()  # type: ignore[call-arg]

    def test_init_with_env_var_project(self):
        """luro.init() should pick up LURO_PROJECT from env."""
        os.environ["LURO_PROJECT"] = "env-project"
        try:
            luro.init()  # type: ignore[call-arg]
            config = get_config()
            assert config.project == "env-project"
        finally:
            del os.environ["LURO_PROJECT"]
