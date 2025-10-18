"""
Configuration for agentic retrieval system.
Uses centralized settings from common.settings.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.settings import get_settings

# Get settings instance
_settings = get_settings()

# OpenAI configuration for agent LLM
OPENAI_API_KEY = _settings.openai_api_key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Set it in .env file or as OPENAI_API_KEY or EXRAG_OPENAI_API_KEY environment variable.")

# Agent LLM model
AGENT_MODEL = _settings.agent_model
TEMPERATURE = _settings.agent_temperature

# ChromaDB configuration (reuse from embeddings)
CHROMA_PERSIST_DIR = _settings.chroma_db_dir

# Retrieval configuration
DEFAULT_TOP_K = _settings.retrieval_top_k
SIMILARITY_THRESHOLD = _settings.retrieval_similarity_threshold

# Exploration configuration
MAX_EXPLORATION_DEPTH = _settings.max_exploration_depth
DEFAULT_EXPLORATION_COUNT = _settings.default_exploration_count

# Agent configuration
MAX_ITERATIONS = _settings.max_agent_iterations
RECURSION_LIMIT = _settings.agent_recursion_limit

# Logging configuration
LOG_LEVEL = _settings.log_level
LOG_FILE = Path(__file__).parent / "retrieval.log"
LOG_FORMAT = _settings.log_format

# Configure logging
def setup_logging():
    """Setup logging configuration for the retrieval system."""
    # Create logger
    logger = logging.getLogger("retrieval")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

