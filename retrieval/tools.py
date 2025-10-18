"""
Retrieval tools for the agentic system.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional
from openai import OpenAI
import chromadb
from chromadb.config import Settings

from .config import CHROMA_PERSIST_DIR, DEFAULT_TOP_K, DEFAULT_EXPLORATION_COUNT, MAX_EXPLORATION_DEPTH, OPENAI_API_KEY

# Get logger
logger = logging.getLogger("retrieval.tools")


class SimpleEmbedder:
    """Simple embedder for OpenAI API calls."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


class RetrievalTools:
    """Collection of tools for semantic retrieval and node exploration."""
    
    def __init__(self, chroma_dir: Path = CHROMA_PERSIST_DIR):
        """Initialize ChromaDB client and embedder."""
        logger.info(f"Initializing RetrievalTools with ChromaDB at: {chroma_dir}")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize simple embedder (same model as used in indexing)
            self.embedder = SimpleEmbedder(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
            logger.info("Initialized OpenAI embedder (text-embedding-3-small)")
            
            self._json_cache: Dict[str, List[Dict[str, Any]]] = {}
            logger.info("RetrievalTools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise
    
    def _load_json_from_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Load JSON file associated with a collection."""
        logger.debug(f"Loading JSON for collection: {collection_name}")
        try:
            # Get collection (no embedding function needed for metadata access)
            collection = self.chroma_client.get_collection(collection_name)
            json_path = collection.metadata.get("json_source")
            
            if not json_path:
                logger.error(f"No JSON source in metadata for collection: {collection_name}")
                raise ValueError(f"No JSON source found in collection metadata for {collection_name}")
            
            # Convert to Path object and resolve relative paths
            json_path_obj = Path(json_path)
            if not json_path_obj.is_absolute():
                # If relative, make it relative to project root (parent of retrieval/)
                project_root = Path(__file__).parent.parent
                json_path_obj = project_root / json_path
                logger.debug(f"Converted relative path to absolute: {json_path_obj}")
            
            # Use the resolved path as cache key
            cache_key = str(json_path_obj)
            
            # Cache the JSON data
            if cache_key not in self._json_cache:
                logger.info(f"Loading JSON file: {json_path_obj}")
                with open(json_path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle various JSON structures
                if isinstance(data, list):
                    nodes = data
                elif isinstance(data, dict):
                    for key in ['structure', 'nodes', 'data', 'items']:
                        if key in data:
                            nodes = data[key]
                            break
                    else:
                        logger.error(f"Unexpected JSON structure in {json_path}")
                        raise ValueError(f"Unexpected JSON structure")
                else:
                    logger.error(f"JSON is not list or dict in {json_path_obj}")
                    raise ValueError(f"Unexpected JSON structure")
                
                self._json_cache[cache_key] = nodes
                logger.info(f"Cached {len(nodes)} nodes from {json_path_obj}")
            else:
                logger.debug(f"Using cached JSON for {cache_key}")
            
            return self._json_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Failed to load JSON for collection {collection_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load JSON for collection {collection_name}: {e}")
    
    def search_by_title(
        self,
        query: str,
        collection_name: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks using title-based embeddings.
        
        Args:
            query: Search query string
            collection_name: Name of the title-indexed collection
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        logger.info(f"Title search: query='{query[:50]}...', collection={collection_name}, top_k={top_k}")
        try:
            # Get collection (no embedding function - we'll provide embeddings manually)
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # Verify this is a title-indexed collection
            index_field = collection.metadata.get("index_field", "text")
            if index_field != "title":
                logger.error(f"Collection {collection_name} indexed on '{index_field}', expected 'title'")
                raise ValueError(f"Collection {collection_name} is indexed on '{index_field}', not 'title'")
            
            logger.debug(f"Generating query embedding for: '{query[:50]}...'")
            # Generate query embedding using the same embedder as indexing
            query_embedding = self.embedder.embed_single(query)
            
            logger.debug(f"Querying title collection with top_k={top_k}")
            # Query the collection with pre-computed embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            chunks = []
            for i in range(len(results['ids'][0])):
                similarity_score = 1 - results['distances'][0][i]
                chunk = {
                    "node_id": results['ids'][0][i],
                    "title": results['documents'][0][i],  # Title is stored as document
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": similarity_score,
                    "source": "title_search"
                }
                chunks.append(chunk)
                logger.debug(f"  Result {i+1}: node_id={chunk['node_id']}, score={similarity_score:.3f}")
            
            logger.info(f"Title search returned {len(chunks)} results")
            return chunks
            
        except Exception as e:
            logger.error(f"Title search failed: {e}", exc_info=True)
            raise RuntimeError(f"Title search failed: {e}")
    
    def search_by_text(
        self,
        query: str,
        collection_name: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks using text-based embeddings.
        
        Args:
            query: Search query string
            collection_name: Name of the text-indexed collection
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        logger.info(f"Text search: query='{query[:50]}...', collection={collection_name}, top_k={top_k}")
        try:
            # Get collection (no embedding function - we'll provide embeddings manually)
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # Verify this is a text-indexed collection
            index_field = collection.metadata.get("index_field", "text")
            if index_field != "text":
                logger.error(f"Collection {collection_name} indexed on '{index_field}', expected 'text'")
                raise ValueError(f"Collection {collection_name} is indexed on '{index_field}', not 'text'")
            
            logger.debug(f"Generating query embedding for: '{query[:50]}...'")
            # Generate query embedding using the same embedder as indexing
            query_embedding = self.embedder.embed_single(query)
            
            logger.debug(f"Querying text collection with top_k={top_k}")
            # Query the collection with pre-computed embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            chunks = []
            for i in range(len(results['ids'][0])):
                similarity_score = 1 - results['distances'][0][i]
                chunk = {
                    "node_id": results['ids'][0][i],
                    "text": results['documents'][0][i],  # Text is stored as document
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": similarity_score,
                    "source": "text_search"
                }
                chunks.append(chunk)
                logger.debug(f"  Result {i+1}: node_id={chunk['node_id']}, score={similarity_score:.3f}")
            
            logger.info(f"Text search returned {len(chunks)} results")
            return chunks
            
        except Exception as e:
            logger.error(f"Text search failed: {e}", exc_info=True)
            raise RuntimeError(f"Text search failed: {e}")
    
    def explore_nodes(
        self,
        node_id: str,
        collection_name: str,
        direction: Literal["up", "down", "both"] = "both",
        count: int = DEFAULT_EXPLORATION_COUNT
    ) -> Dict[str, Any]:
        """
        Explore nodes above and/or below a given node in the document structure.
        
        Args:
            node_id: ID of the node to explore around
            collection_name: Name of the collection (to find associated JSON)
            direction: Direction to explore ("up", "down", or "both")
            count: Number of nodes to retrieve in each direction
            
        Returns:
            Dictionary containing the target node and surrounding nodes
        """
        logger.info(f"Exploring nodes: node_id={node_id}, direction={direction}, count={count}")
        try:
            # Validate count
            if count < 1 or count > MAX_EXPLORATION_DEPTH:
                logger.error(f"Invalid count: {count}, must be 1-{MAX_EXPLORATION_DEPTH}")
                raise ValueError(f"Count must be between 1 and {MAX_EXPLORATION_DEPTH}")
            
            # Load the JSON structure
            nodes = self._load_json_from_collection(collection_name)
            logger.debug(f"Loaded {len(nodes)} nodes from collection")
            
            # Find the target node index
            target_idx = None
            for idx, node in enumerate(nodes):
                if node.get("node_id") == node_id:
                    target_idx = idx
                    break
            
            if target_idx is None:
                logger.error(f"Node {node_id} not found in JSON structure")
                raise ValueError(f"Node {node_id} not found in JSON structure")
            
            logger.debug(f"Found target node at index {target_idx}")
            
            # Collect surrounding nodes
            result = {
                "target_node": nodes[target_idx],
                "nodes_above": [],
                "nodes_below": []
            }
            
            # Get nodes above
            if direction in ["up", "both"]:
                start_idx = max(0, target_idx - count)
                result["nodes_above"] = nodes[start_idx:target_idx]
                logger.debug(f"Retrieved {len(result['nodes_above'])} nodes above")
            
            # Get nodes below
            if direction in ["down", "both"]:
                end_idx = min(len(nodes), target_idx + count + 1)
                result["nodes_below"] = nodes[target_idx + 1:end_idx]
                logger.debug(f"Retrieved {len(result['nodes_below'])} nodes below")
            
            # Add position information
            result["position_info"] = {
                "target_index": target_idx,
                "total_nodes": len(nodes),
                "can_go_up": target_idx > 0,
                "can_go_down": target_idx < len(nodes) - 1
            }
            
            logger.info(f"Exploration complete: {len(result['nodes_above'])} above, {len(result['nodes_below'])} below")
            return result
            
        except Exception as e:
            logger.error(f"Node exploration failed: {e}", exc_info=True)
            raise RuntimeError(f"Node exploration failed: {e}")
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections with their metadata.
        
        Returns:
            List of collection information dictionaries
        """
        logger.info("Listing all collections")
        try:
            collections = self.chroma_client.list_collections()
            logger.debug(f"Found {len(collections)} collections")
            
            collection_info = []
            for coll in collections:
                info = {
                    "name": coll.name,
                    "count": coll.count(),
                    "metadata": coll.metadata
                }
                collection_info.append(info)
                logger.debug(f"  Collection: {coll.name}, count={info['count']}")
            
            logger.info(f"Retrieved info for {len(collection_info)} collections")
            return collection_info
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}", exc_info=True)
            raise RuntimeError(f"Failed to list collections: {e}")

