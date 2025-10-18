"""
ChromaDB collection creation and indexing logic.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL
from metadata_extractor import extract_metadata, is_valid_node
from embedder import Embedder


def generate_collection_name(json_path: Path) -> str:
    """
    Generate a unique, valid collection name from JSON file path.
    
    ChromaDB collection names must:
    - Start and end with alphanumeric
    - Contain only alphanumeric, underscores, hyphens
    - Be 3-63 characters long
    """
    # Use filename stem as base
    base_name = json_path.stem
    
    # Clean name: replace invalid chars with underscore
    clean_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in base_name
    )
    
    # Add short hash for uniqueness
    path_hash = hashlib.md5(str(json_path.absolute()).encode()).hexdigest()[:8]
    
    collection_name = f"{clean_name}_{path_hash}"
    
    # Ensure valid length
    if len(collection_name) > 63:
        collection_name = collection_name[:55] + path_hash
    
    return collection_name.lower()


class VectorIndexer:
    """Manages ChromaDB indexing for JSON tree files."""
    
    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        """Initialize ChromaDB client with persistent storage."""
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy initialization of embedder (requires API key)."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder
    
    def load_json_nodes(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load and validate nodes from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle various JSON structures
        if isinstance(data, list):
            nodes = data
        elif isinstance(data, dict):
            # Try common keys: structure, nodes, data, items
            for key in ['structure', 'nodes', 'data', 'items']:
                if key in data:
                    nodes = data[key]
                    break
            else:
                raise ValueError(f"Unexpected JSON structure in {json_path}. Expected array or object with 'structure'/'nodes' key")
        else:
            raise ValueError(f"Unexpected JSON structure in {json_path}")
        
        print(f"Loaded {len(nodes)} nodes from {json_path.name}")
        
        # Filter valid nodes
        valid_nodes = [n for n in nodes if is_valid_node(n)]
        filtered_count = len(nodes) - len(valid_nodes)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} empty/trivial nodes")
        
        return valid_nodes
    
    def create_or_reset_collection(self, collection_name: str, reset: bool = False, index_field: str = "text", json_path: str = None):
        """Create or get existing collection."""
        if reset:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection didn't exist
        
        metadata = {
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": str(EMBEDDING_DIMENSION),
            "index_field": index_field,
        }
        
        # Store JSON source path for node lookup (use absolute path)
        if json_path:
            metadata["json_source"] = str(Path(json_path).absolute())
        
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        return collection
    
    def index_nodes(
        self,
        nodes: List[Dict[str, Any]],
        collection_name: str,
        reset: bool = False,
        index_field: str = "text",
        json_path: Path = None,
    ) -> Dict[str, Any]:
        """
        Index all nodes into ChromaDB collection with batch embedding.
        
        Args:
            nodes: List of node dictionaries
            collection_name: Name for ChromaDB collection
            reset: If True, delete existing collection first
            index_field: Which field to embed (text, title, summary, etc.)
            
        Returns:
            Statistics about the indexing process
        """
        collection = self.create_or_reset_collection(collection_name, reset=reset, index_field=index_field, json_path=json_path)
        
        stats = {
            "total_nodes": len(nodes),
            "batches": 0,
            "indexed": 0,
            "failed": 0,
            "skipped": 0,
            "index_field": index_field,
        }
        
        print(f"\nIndexing {len(nodes)} nodes into collection '{collection_name}'")
        print(f"Index field: {index_field}")
        print(f"Batch size: {BATCH_SIZE}")
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(nodes), BATCH_SIZE), desc="Embedding batches"):
            batch_nodes = nodes[i:i + BATCH_SIZE]
            
            try:
                # Extract content from the specified field
                texts = []
                valid_nodes = []
                valid_ids = []
                valid_metadatas = []
                
                for j, node in enumerate(batch_nodes):
                    # Get content from the specified field
                    content = node.get(index_field, "")
                    
                    # Skip nodes without the specified field or empty content
                    if not content or not content.strip():
                        stats["skipped"] += 1
                        continue
                    
                    texts.append(content)
                    valid_nodes.append(node)
                    valid_ids.append(node.get("node_id", f"node_{i + j}"))
                    valid_metadatas.append(extract_metadata(node))
                
                # Skip batch if no valid nodes
                if not texts:
                    continue
                
                # Generate embeddings
                embeddings = self.embedder.embed_batch(texts)
                
                # Add to ChromaDB (store original text field for display)
                collection.add(
                    ids=valid_ids,
                    embeddings=embeddings,
                    documents=texts,  # Store the indexed field content
                    metadatas=valid_metadatas,
                )
                
                stats["batches"] += 1
                stats["indexed"] += len(valid_nodes)
                
            except Exception as e:
                print(f"\n  Error indexing batch {i//BATCH_SIZE + 1}: {e}")
                stats["failed"] += len(batch_nodes)
        
        return stats
    
    def index_json_file(
        self,
        json_path: Path,
        collection_name: str = None,
        reset: bool = False,
        index_field: str = "text",
    ) -> Dict[str, Any]:
        """
        Complete pipeline: load JSON, create collection, index nodes.
        
        Args:
            json_path: Path to JSON tree file
            collection_name: Optional custom collection name
            reset: If True, delete existing collection first
            index_field: Which field to embed (text, title, summary, etc.)
            
        Returns:
            Statistics dictionary
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Generate collection name if not provided
        if collection_name is None:
            collection_name = generate_collection_name(json_path)
        
        print(f"\n{'='*60}")
        print("Vectorization Pipeline")
        print(f"{'='*60}")
        print(f"Source: {json_path}")
        print(f"Collection: {collection_name}")
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Index Field: {index_field}")
        print(f"{'='*60}\n")
        
        # Load nodes
        nodes = self.load_json_nodes(json_path)
        
        # Index with embeddings
        stats = self.index_nodes(nodes, collection_name, reset=reset, index_field=index_field, json_path=json_path)
        
        # Add metadata
        stats["collection_name"] = collection_name
        stats["json_file"] = str(json_path)
        stats["model"] = EMBEDDING_MODEL
        stats["index_field"] = index_field
        
        return stats
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception as e:
            return {"error": str(e)}

