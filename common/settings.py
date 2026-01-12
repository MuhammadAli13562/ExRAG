"""
Centralized configuration using Pydantic BaseSettings.
Loads defaults and overrides from environment variables and .env files.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file from project root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings(BaseSettings):
    """Global settings for the ExRAG pipeline."""
    
    # ===== Project Paths =====
    project_root: Path = Path(__file__).parent.parent
    md_dir: Path = project_root / "md"
    results_dir: Path = project_root / "results"
    embeddings_dir: Path = project_root / "embeddings"
    chroma_db_dir: Path = embeddings_dir / "chroma_db"
    
    # ===== OpenAI Configuration =====
    openai_api_key: str = ""
    
    # ===== Embedding Configuration =====
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_provider: str = "openai"
    embedding_batch_size: int = 100
    embedding_max_retries: int = 3
    embedding_retry_delay: int = 2  # seconds
    
    # ===== Agent/Retrieval Configuration =====
    agent_model: str = "gpt-4o"
    agent_temperature: float = 0.0
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.5
    max_agent_iterations: int = 3
    agent_recursion_limit: int = 25
    max_exploration_depth: int = 10
    default_exploration_count: int = 3
    
    # ===== Logging =====
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: Optional[Path] = None
    
    # ===== Build Configuration =====
    tree_mode: str = "full"  # "simple" or "full"
    default_collection_prefix: str = "exrag"
    
    model_config = SettingsConfigDict(
        env_prefix="EXRAG_",  # Accept both EXRAG_* and standard env vars
        case_sensitive=False,
        extra="ignore",
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default log file if not specified
        if self.log_to_file and self.log_file is None:
            self.log_file = self.project_root / "exrag.log"
        
        # Validate API key if provider is OpenAI
        if self.embedding_provider == "openai" and not self.openai_api_key:
            # Try to get from standard OPENAI_API_KEY env var
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    
    def validate_openai_key(self) -> bool:
        """Check if OpenAI API key is set."""
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to check if we're ready to use OpenAI
def check_openai_available() -> tuple[bool, str]:
    """
    Check if OpenAI is properly configured.
    
    Returns:
        (is_available, message)
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return False, "OPENAI_API_KEY not set in environment or .env file"
    return True, "OpenAI API key configured"

