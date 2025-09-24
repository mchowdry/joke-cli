"""
Unit tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from joke_cli import config


class TestConfig:
    """Test cases for configuration module."""

    def test_default_constants(self):
        """Test that default constants are properly defined."""
        assert config.DEFAULT_MODEL_ID == "amazon.titan-text-express-v1"
        assert config.DEFAULT_AWS_REGION == "us-east-1"
        assert config.MAX_TOKENS == 200
        assert config.TEMPERATURE == 0.7
        assert config.TOP_P == 0.9

    def test_available_categories(self):
        """Test that joke categories are properly defined."""
        expected_categories = ["general", "programming", "dad-jokes", "puns", "clean"]
        assert config.AVAILABLE_CATEGORIES == expected_categories

    def test_feedback_configuration(self):
        """Test feedback-related configuration."""
        assert config.FEEDBACK_RATING_MIN == 1
        assert config.FEEDBACK_RATING_MAX == 5
        assert config.FEEDBACK_STORAGE_FILENAME == "joke_feedback.json"

    def test_get_feedback_storage_dir(self):
        """Test feedback storage directory creation."""
        storage_dir = config.get_feedback_storage_dir()
        assert isinstance(storage_dir, Path)
        assert storage_dir.name == ".joke_cli"
        assert storage_dir.exists()  # Should be created if it doesn't exist

    @patch.dict(os.environ, {"AWS_PROFILE": "test-profile"})
    def test_get_aws_profile_with_env_var(self):
        """Test getting AWS profile from environment variable."""
        profile = config.get_aws_profile()
        assert profile == "test-profile"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_aws_profile_without_env_var(self):
        """Test getting AWS profile when environment variable is not set."""
        profile = config.get_aws_profile()
        assert profile is None

    @patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_aws_region_with_env_var(self):
        """Test getting AWS region from environment variable."""
        region = config.get_aws_region()
        assert region == "us-west-2"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_aws_region_without_env_var(self):
        """Test getting AWS region when environment variable is not set."""
        region = config.get_aws_region()
        assert region == config.DEFAULT_AWS_REGION

    def test_error_messages(self):
        """Test that error messages are properly defined."""
        assert "no_credentials" in config.ERROR_MESSAGES
        assert "access_denied" in config.ERROR_MESSAGES
        assert "invalid_category" in config.ERROR_MESSAGES
        assert "network_error" in config.ERROR_MESSAGES
        assert "service_unavailable" in config.ERROR_MESSAGES
        assert "rate_limit" in config.ERROR_MESSAGES

    def test_cli_configuration(self):
        """Test CLI-related configuration."""
        assert config.CLI_COMMAND_NAME == "joke"
        assert config.CLI_DESCRIPTION == "Generate jokes using AWS Bedrock AI models"