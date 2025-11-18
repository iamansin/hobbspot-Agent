"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError
from unittest.mock import patch
import os


def test_settings_with_all_required_env_vars(monkeypatch):
    """Test Settings loads successfully with all required environment variables."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set all required environment variables
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    from app.config import Settings
    settings = Settings()
    
    # Verify required fields are loaded
    assert settings.appwrite_endpoint == "https://cloud.appwrite.io/v1"
    assert settings.appwrite_project_id == "test_project"
    assert settings.appwrite_api_key == "test_api_key"
    assert settings.appwrite_database_id == "test_db"
    assert settings.appwrite_collection_id == "test_collection"
    assert settings.openai_api_key == "test_openai_key"
    assert settings.gemini_api_key == "test_gemini_key"
    assert settings.brave_api_key == "test_brave_key"


def test_settings_default_values(monkeypatch):
    """Test Settings uses correct default values for optional fields."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set only required environment variables
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    from app.config import Settings
    settings = Settings()
    
    # Verify default values
    assert settings.previous_message_context_length == 10
    assert settings.overlap_count == 5
    assert settings.cache_ttl_seconds == 600
    assert settings.default_llm_provider == "openai"
    assert settings.cache_directory == "./cache"
    assert settings.log_level == "INFO"
    assert settings.log_rotation == "100 MB"
    assert settings.log_retention == "30 days"


def test_settings_custom_optional_values(monkeypatch):
    """Test Settings correctly loads custom values for optional fields."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set all environment variables including optional ones
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
        "PREVIOUS_MESSAGE_CONTEXT_LENGTH": "20",
        "OVERLAP_COUNT": "10",
        "CACHE_TTL_SECONDS": "1200",
        "DEFAULT_LLM_PROVIDER": "gemini",
        "CACHE_DIRECTORY": "/custom/cache",
        "LOG_LEVEL": "DEBUG",
        "LOG_ROTATION": "50 MB",
        "LOG_RETENTION": "60 days",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    from app.config import Settings
    settings = Settings()
    
    # Verify custom values are loaded
    assert settings.previous_message_context_length == 20
    assert settings.overlap_count == 10
    assert settings.cache_ttl_seconds == 1200
    assert settings.default_llm_provider == "gemini"
    assert settings.cache_directory == "/custom/cache"
    assert settings.log_level == "DEBUG"
    assert settings.log_rotation == "50 MB"
    assert settings.log_retention == "60 days"


def test_settings_missing_required_field_raises_validation_error(monkeypatch):
    """Test Settings raises ValidationError when required fields are missing."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set only some required environment variables (missing OPENAI_API_KEY)
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    from app.config import Settings
    
    # Should raise ValidationError for missing OPENAI_API_KEY
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    # Verify the error mentions the missing field
    assert "openai_api_key" in str(exc_info.value).lower()


def test_get_settings_singleton_pattern(monkeypatch):
    """Test get_settings returns the same instance (singleton pattern)."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set required environment variables
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    # Get settings twice
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Verify they are the same instance
    assert settings1 is settings2


def test_settings_validation_error_for_invalid_types(monkeypatch):
    """Test Settings raises ValidationError for invalid data types."""
    # Clear the lru_cache before test
    from app.config import get_settings
    get_settings.cache_clear()
    
    # Set environment variables with invalid type for integer field
    env_vars = {
        "APPWRITE_ENDPOINT": "https://cloud.appwrite.io/v1",
        "APPWRITE_PROJECT_ID": "test_project",
        "APPWRITE_API_KEY": "test_api_key",
        "APPWRITE_DATABASE_ID": "test_db",
        "APPWRITE_COLLECTION_ID": "test_collection",
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "BRAVE_API_KEY": "test_brave_key",
        "PREVIOUS_MESSAGE_CONTEXT_LENGTH": "not_a_number",  # Invalid type
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    from app.config import Settings
    
    # Should raise ValidationError for invalid type
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    # Verify the error mentions the field with invalid type
    assert "previous_message_context_length" in str(exc_info.value).lower()
