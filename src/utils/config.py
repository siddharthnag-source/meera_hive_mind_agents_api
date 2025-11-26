"""Configuration management for Meera OS."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gemini API
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    # gemini_model: str = Field(default="gemini-2.5-pro", env="GEMINI_MODEL")
    # gemini_model: str = Field(default="gemini-2.0-flash-lite", env="GEMINI_MODEL")
    gemini_model: str = Field(default="gemini-flash-latest", env="GEMINI_MODEL")
    
    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    mongodb_database: str = Field(default="meera_os", env="MONGODB_DATABASE")
    mongodb_memory_collection: str = Field(
        default="memory_nodes", env="MONGODB_MEMORY_COLLECTION"
    )
    mongodb_user_identity_collection: str = Field(
        default="user_identities", env="MONGODB_USER_IDENTITY_COLLECTION"
    )
    
    # Vector DB
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(
        default="memory_embeddings", env="CHROMA_COLLECTION_NAME"
    )
    
    # System
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_personal_memories: int = Field(default=3, env="MAX_PERSONAL_MEMORIES")
    max_hive_mind_memories: int = Field(default=3, env="MAX_HIVE_MIND_MEMORIES")
    embedding_model: str = Field(default="text-embedding-004", env="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ConfigLoader:
    """Loads configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config is None:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        config = self.load()
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # existing fields...
    supabase_url: str
    supabase_service_role_key: str

    class Config:
        env_prefix = ""  # or whatever you already use
        case_sensitive = False


# Global settings instance
settings = Settings()
config_loader = ConfigLoader()

