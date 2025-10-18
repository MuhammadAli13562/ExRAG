"""
Configuration for embedding generation and vector indexing.
Uses centralized settings from common.settings.
"""
import sys
from pathlib import Path

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.settings import get_settings

# Get settings instance
_settings = get_settings()

# Embedding model configuration
EMBEDDING_MODEL = _settings.embedding_model
EMBEDDING_DIMENSION = _settings.embedding_dimension
EMBEDDING_PROVIDER = _settings.embedding_provider

# ChromaDB configuration
CHROMA_PERSIST_DIR = _settings.chroma_db_dir

# Batch processing
BATCH_SIZE = _settings.embedding_batch_size
MAX_RETRIES = _settings.embedding_max_retries
RETRY_DELAY = _settings.embedding_retry_delay

# Metadata extraction patterns
PAGE_PATTERN = r'p\.(\d+)'
CHAPTER_PATTERN = r'^(\d+)(?:\.|:|\s|$)'
SECTION_PATTERN = r'^(\d+\.\d+(?:\.\d+)?)'
ANCHOR_KEYWORDS = [
    "Experiment",
    "Discussion",
    "Summary",
    "Example",
    "DSE exam",
    "DSE goal",
    "Flipped classroom",
    "Investigation",
    "Activity",
    "Exercise",
]

# OpenAI API configuration
OPENAI_API_KEY = _settings.openai_api_key
if not OPENAI_API_KEY:
    import warnings
    warnings.warn("OPENAI_API_KEY not found in environment variables. Set it in .env file or as OPENAI_API_KEY or EXRAG_OPENAI_API_KEY environment variable.")

