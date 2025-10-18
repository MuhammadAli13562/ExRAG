"""
Configuration for embedding generation and vector indexing.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
EMBEDDING_PROVIDER = "openai"  # Currently only OpenAI

# ChromaDB configuration
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)

# Batch processing
BATCH_SIZE = 100  # Number of nodes to embed at once
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

