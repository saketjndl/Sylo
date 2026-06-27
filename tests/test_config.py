"""Tests for Sylo SDK configuration."""

from __future__ import annotations

import os

import pytest

import sylo
from sylo.config import SyloConfig, get_config, reset_config, set_config
from sylo.exceptions import SyloConfigError


class TestSyloConfig:
    """Tests for SyloConfig model validation."""

    def test_minimal_config(self):
        """Config with just project name should work."""
        config = SyloConfig(project="my-project")
        assert config.project == "my-project"
        assert config.api_key is None
        assert config.environment == "development"
        assert config.storage == "local"

    def test_full_config(self):
        """All fields should be settable."""
        config = SyloConfig(
            project="my-project",
            api_key="sylo_test_key",
            environment="production",
            storage="redis",
            redis_url="redis://myhost:6380",
        )
        assert config.project == "my-project"
        assert config.api_key == "sylo_test_key"
        assert config.environment == "production"
        assert config.storage == "redis"
        assert config.redis_url == "redis://myhost:6380"

    def test_invalid_environment_rejected(self):
        """Invalid environment values should raise validation error."""
        with pytest.raises(Exception):
            SyloConfig(project="test", environment="invalid")

    def test_invalid_storage_rejected(self):
        """Invalid storage values should raise validation error."""
        with pytest.raises(Exception):
            SyloConfig(project="test", storage="mongodb")

    def test_env_var_fallback(self):
        """Environment variables should fill in missing config values."""
        os.environ["SYLO_PROJECT"] = "env-project"
        os.environ["SYLO_API_KEY"] = "sylo_env_key"
        os.environ["SYLO_ENVIRONMENT"] = "staging"
        try:
            config = SyloConfig()  # type: ignore[call-arg]
            assert config.project == "env-project"
            assert config.api_key == "sylo_env_key"
            assert config.environment == "staging"
        finally:
            del os.environ["SYLO_PROJECT"]
            del os.environ["SYLO_API_KEY"]
            del os.environ["SYLO_ENVIRONMENT"]

    def test_programmatic_overrides_env(self):
        """Programmatic values should take precedence over env vars."""
        os.environ["SYLO_PROJECT"] = "env-project"
        try:
            config = SyloConfig(project="programmatic-project")
            assert config.project == "programmatic-project"
        finally:
            del os.environ["SYLO_PROJECT"]

    def test_is_development(self):
        """is_development property should reflect environment."""
        config = SyloConfig(project="test", environment="development")
        assert config.is_development is True
        assert config.is_production is False

    def test_is_production(self):
        """is_production property should reflect environment."""
        config = SyloConfig(project="test", environment="production")
        assert config.is_production is True
        assert config.is_development is False

    def test_has_cloud(self):
        """has_cloud should be True only when api_key is set."""
        config_no_key = SyloConfig(project="test")
        assert config_no_key.has_cloud is False

        config_with_key = SyloConfig(project="test", api_key="sylo_xxx")
        assert config_with_key.has_cloud is True


class TestGlobalConfig:
    """Tests for global config singleton."""

    def test_get_config_before_init_raises(self):
        """Accessing config before init should raise SyloConfigError."""
        with pytest.raises(SyloConfigError, match="not initialized"):
            get_config()

    def test_set_and_get_config(self):
        """Setting config should make it retrievable."""
        config = SyloConfig(project="test")
        set_config(config)
        assert get_config().project == "test"

    def test_reset_config(self):
        """Resetting config should make get_config raise again."""
        set_config(SyloConfig(project="test"))
        reset_config()
        with pytest.raises(SyloConfigError):
            get_config()


class TestSyloInit:
    """Tests for the sylo.init() function."""

    def test_basic_init(self):
        """sylo.init() should set global config."""
        sylo.init(project="my-project")
        config = get_config()
        assert config.project == "my-project"
        assert config.environment == "development"
        assert config.storage == "local"

    def test_init_with_all_options(self):
        """sylo.init() should accept all config options."""
        sylo.init(
            project="my-project",
            api_key="sylo_key",
            environment="production",
            storage="local",
        )
        config = get_config()
        assert config.project == "my-project"
        assert config.api_key == "sylo_key"
        assert config.environment == "production"

    def test_init_without_project_raises(self):
        """sylo.init() without project should raise SyloConfigError."""
        with pytest.raises(SyloConfigError):
            sylo.init()  # type: ignore[call-arg]

    def test_init_with_env_var_project(self):
        """sylo.init() should pick up SYLO_PROJECT from env."""
        os.environ["SYLO_PROJECT"] = "env-project"
        try:
            sylo.init()  # type: ignore[call-arg]
            config = get_config()
            assert config.project == "env-project"
        finally:
            del os.environ["SYLO_PROJECT"]
