"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Context management
    previous_message_context_length: int = 10
    overlap_count: int = 5
    cache_ttl_seconds: int = 600  # 10 minutes
    
    # Appwrite
    appwrite_endpoint: str
    appwrite_project_id: str
    appwrite_api_key: str
    appwrite_database_id: str
    appwrite_collection_id: str
    
    # LLM Providers
    openai_api_key: str
    gemini_api_key: str
    default_llm_provider: str = "openai"
    
    # Brave Search
    brave_api_key: str
    
    # DiskCache
    cache_directory: str = "./cache"
    
    # Logging
    log_level: str = "INFO"
    log_rotation: str = "100 MB"
    log_retention: str = "30 days"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Uses lru_cache to ensure only one Settings instance is created.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
