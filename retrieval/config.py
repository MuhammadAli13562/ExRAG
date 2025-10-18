"""
Configuration for agentic retrieval system.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI configuration for agent LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Agent LLM model
AGENT_MODEL = "gpt-4o"  # Using GPT-4o for the agent
TEMPERATURE = 0  # Deterministic for consistency

# ChromaDB configuration (reuse from embeddings)
CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "embeddings" / "chroma_db"

# Retrieval configuration
DEFAULT_TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score

# Exploration configuration
MAX_EXPLORATION_DEPTH = 10  # Maximum nodes to explore in one direction
DEFAULT_EXPLORATION_COUNT = 3  # Default number of nodes to explore

# Agent configuration
MAX_ITERATIONS = 15  # Maximum number of agent iterations
RECURSION_LIMIT = 25  # LangGraph recursion limit

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = Path(__file__).parent / "retrieval.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# LangSmith tracing configuration (optional)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", None)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "pageindex-retrieval")

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

