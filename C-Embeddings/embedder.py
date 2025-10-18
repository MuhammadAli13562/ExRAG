"""
Embedding generation using OpenAI API.
"""
import time
from typing import List
from openai import OpenAI
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    MAX_RETRIES,
    RETRY_DELAY,
)


class Embedder:
    """Handles embedding generation with retry logic."""
    
    def __init__(self):
        if EMBEDDING_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = EMBEDDING_MODEL
        else:
            raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                
                # Extract embeddings in original order
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to embed batch after {MAX_RETRIES} attempts: {e}")
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

