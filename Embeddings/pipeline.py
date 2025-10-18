"""
Unified embedding pipeline interface.
Orchestrates node loading, embedding, and indexing with clean function APIs.
"""
import json
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any

# Import local modules
from .indexer import VectorIndexer, generate_collection_name
from .node_loader import build_node_lookup

# Import common models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.models import EmbeddingConfig, EmbeddingRunResult, IndexStats
from common.settings import get_settings
from common.logging_config import get_logger


logger = get_logger("exrag.embeddings")


def load_nodes_from_tree(tree_json: Path) -> List[Dict[str, Any]]:
    """
    Load all nodes from a tree JSON file (flattened).
    
    Args:
        tree_json: Path to tree JSON file
    
    Returns:
        List of node dictionaries
    """
    tree_json = Path(tree_json)
    
    if not tree_json.exists():
        raise FileNotFoundError(f"Tree JSON not found: {tree_json}")
    
    logger.info(f"Loading nodes from {tree_json.name}")
    
    with open(tree_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten tree structure
    def flatten_tree(nodes, result_list):
        for node in nodes:
            result_list.append({
                'node_id': node.get('node_id'),
                'title': node.get('title', ''),
                'text': node.get('text', ''),
                'line_num': node.get('line_num'),
            })
            if 'nodes' in node and node['nodes']:
                flatten_tree(node['nodes'], result_list)
    
    flat_nodes = []
    structure = data.get('structure', [])
    flatten_tree(structure, flat_nodes)
    
    logger.info(f"Loaded {len(flat_nodes)} nodes")
    return flat_nodes


def build_embedding_indexes(
    tree_json: Path,
    chroma_dir: Optional[Path] = None,
    config: Optional[EmbeddingConfig] = None,
    which: Literal["titles", "texts", "both"] = "both",
    collection_prefix: Optional[str] = None,
    reset: bool = False,
) -> EmbeddingRunResult:
    """
    Build embedding indexes from a tree JSON file.
    
    This is the main entry point for the embedding pipeline. It:
    - Loads nodes from the tree JSON
    - Creates ChromaDB collections for title and/or text indexes
    - Embeds and indexes the specified fields
    - Returns detailed statistics
    
    Args:
        tree_json: Path to tree JSON file
        chroma_dir: ChromaDB directory (uses settings default if None)
        config: Embedding configuration (uses defaults if None)
        which: Which indexes to build - "titles", "texts", or "both"
        collection_prefix: Custom prefix for collection names
        reset: If True, delete existing collections before indexing
    
    Returns:
        EmbeddingRunResult with statistics
    
    Raises:
        FileNotFoundError: If tree_json doesn't exist
        ValueError: If embedding fails
    """
    tree_json = Path(tree_json)
    
    if not tree_json.exists():
        raise FileNotFoundError(f"Tree JSON not found: {tree_json}")
    
    # Load settings and config
    settings = get_settings()
    if chroma_dir is None:
        chroma_dir = settings.chroma_db_dir
    else:
        chroma_dir = Path(chroma_dir)
    
    if config is None:
        config = EmbeddingConfig(
            model=settings.embedding_model,
            provider=settings.embedding_provider,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            retry_delay=settings.embedding_retry_delay,
        )
    
    # Check API key
    if not settings.validate_openai_key():
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    
    logger.info("="*60)
    logger.info("Embedding Pipeline")
    logger.info("="*60)
    logger.info(f"Source: {tree_json}")
    logger.info(f"ChromaDB: {chroma_dir}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Building: {which}")
    logger.info("="*60)
    
    # Load nodes
    nodes = load_nodes_from_tree(tree_json)
    
    if not nodes:
        raise ValueError(f"No nodes found in {tree_json}")
    
    # Initialize indexer
    indexer = VectorIndexer(persist_dir=chroma_dir)
    
    title_stats = None
    text_stats = None
    total_indexed = 0
    
    # Generate collection names
    if collection_prefix:
        base_name = collection_prefix
    else:
        base_name = generate_collection_name(tree_json)
    
    # Build title index
    if which in ["titles", "both"]:
        logger.info(f"\nBuilding title index...")
        title_collection = f"{base_name}_titles"
        
        stats_dict = indexer.index_nodes(
            nodes=nodes,
            collection_name=title_collection,
            reset=reset,
            index_field="title",
            json_path=tree_json,
        )
        
        title_stats = IndexStats(
            collection_name=title_collection,
            index_field="title",
            total_nodes=stats_dict["total_nodes"],
            indexed=stats_dict["indexed"],
            skipped=stats_dict["skipped"],
            failed=stats_dict["failed"],
            batches=stats_dict["batches"],
            model=config.model,
        )
        total_indexed += title_stats.indexed
        
        logger.info(f"✓ Title index: {title_stats.indexed}/{title_stats.total_nodes} indexed")
    
    # Build text index
    if which in ["texts", "both"]:
        logger.info(f"\nBuilding text index...")
        text_collection = f"{base_name}_texts"
        
        stats_dict = indexer.index_nodes(
            nodes=nodes,
            collection_name=text_collection,
            reset=reset,
            index_field="text",
            json_path=tree_json,
        )
        
        text_stats = IndexStats(
            collection_name=text_collection,
            index_field="text",
            total_nodes=stats_dict["total_nodes"],
            indexed=stats_dict["indexed"],
            skipped=stats_dict["skipped"],
            failed=stats_dict["failed"],
            batches=stats_dict["batches"],
            model=config.model,
        )
        total_indexed += text_stats.indexed
        
        logger.info(f"✓ Text index: {text_stats.indexed}/{text_stats.total_nodes} indexed")
    
    logger.info("="*60)
    logger.info(f"Total indexed: {total_indexed} vectors")
    logger.info("="*60)
    
    return EmbeddingRunResult(
        tree_json=tree_json,
        chroma_dir=chroma_dir,
        title_stats=title_stats,
        text_stats=text_stats,
        total_indexed=total_indexed,
        config=config,
    )


def list_collections(chroma_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all collections in ChromaDB.
    
    Args:
        chroma_dir: ChromaDB directory (uses settings default if None)
    
    Returns:
        List of collection info dictionaries
    """
    settings = get_settings()
    if chroma_dir is None:
        chroma_dir = settings.chroma_db_dir
    
    indexer = VectorIndexer(persist_dir=chroma_dir)
    collection_names = indexer.list_collections()
    
    collections = []
    for name in collection_names:
        info = indexer.get_collection_info(name)
        collections.append(info)
    
    return collections


def delete_collection(collection_name: str, chroma_dir: Optional[Path] = None) -> bool:
    """
    Delete a collection from ChromaDB.
    
    Args:
        collection_name: Name of collection to delete
        chroma_dir: ChromaDB directory (uses settings default if None)
    
    Returns:
        True if deleted successfully
    """
    settings = get_settings()
    if chroma_dir is None:
        chroma_dir = settings.chroma_db_dir
    
    indexer = VectorIndexer(persist_dir=chroma_dir)
    return indexer.delete_collection(collection_name)

