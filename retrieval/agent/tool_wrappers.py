"""
Tool wrappers for the retrieval agent.

Uses dependency injection pattern - tools are created via factory function
with a RetrievalTools instance passed in, eliminating global state.
"""
import json
import logging
from typing import List

from langchain_core.tools import tool

from ..tools import RetrievalTools

logger = logging.getLogger(__name__)


def create_tool_wrappers(
    retrieval_tools_instance: RetrievalTools,
    deep_log_enabled: bool = False
) -> List:
    """
    Create tool wrappers with injected RetrievalTools instance.
    
    Args:
        retrieval_tools_instance: The RetrievalTools instance to use
        deep_log_enabled: Whether to enable deep logging of tool inputs/outputs
    
    Returns:
        List of tool functions for use with LangChain agents
    """
    
    @tool
    def search_by_title(query: str, collection_name: str, top_k: int = 5) -> str:
        """
        Search for relevant chunks using title-based semantic search.
        Use this when you want to find sections based on their titles or headings.
        
        Args:
            query: The search query describing what you're looking for
            collection_name: Name of the title-indexed collection to search
            top_k: Number of top results to return (default: 5)
        
        Returns:
            JSON string containing matching chunks with node_ids, titles, and similarity scores
        """
        # Title search is intentionally disabled (it can be misleading for this corpus).
        logger.warning("[TOOL] search_by_title is disabled; use search_by_text instead")
        return "Error: search_by_title is disabled. Use search_by_text for semantic retrieval."

    @tool
    def search_by_text(query: str, collection_name: str, top_k: int = 5) -> str:
        """
        Search for relevant chunks using text-based semantic search.
        Use this when you want to find content based on the actual text/content of sections.
        
        Args:
            query: The search query describing what you're looking for
            collection_name: Name of the text-indexed collection to search
            top_k: Number of top results to return (default: 5)
        
        Returns:
            JSON string containing matching chunks with node_ids, text content, and similarity scores
        """
        logger.info(f"[TOOL] search_by_text called with query='{query[:50]}...', top_k={top_k}")
        
        if deep_log_enabled:
            logger.info("="*80)
            logger.info("[DEEP LOG] search_by_text INPUT:")
            logger.info(f"  query: {query}")
            logger.info(f"  collection_name: {collection_name}")
            logger.info(f"  top_k: {top_k}")
            logger.info("="*80)
        
        try:
            results = retrieval_tools_instance.search_by_text(query, collection_name, top_k)
            logger.info(f"[TOOL] search_by_text returned {len(results)} results")
            
            if deep_log_enabled:
                logger.info("="*80)
                logger.info(f"[DEEP LOG] search_by_text OUTPUT ({len(results)} results):")
                for i, result in enumerate(results, 1):
                    logger.info(f"\n  Result {i}:")
                    logger.info(f"    node_id: {result.get('node_id', 'N/A')}")
                    logger.info(f"    text: {result.get('text', 'N/A')[:200]}...")
                    logger.info(f"    similarity_score: {result.get('similarity_score', 'N/A')}")
                    if 'metadata' in result:
                        logger.info(f"    metadata: {result['metadata']}")
                logger.info("="*80)
            
            # Add citation reminder to output
            output = {
                "results": results,
                "CITATION_REMINDER": "🚨 MANDATORY: Use the node_id from these results to cite in your final answer. Format: [Node XXXX]"
            }
            return json.dumps(output, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] search_by_text failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def explore_nodes(node_id: str, collection_name: str, direction: str = "both", count: int = 3) -> str:
        """
        Explore nodes above and/or below a specific node to get surrounding context.
        Use this when you find a relevant chunk and want to see what comes before or after it.
        
        Args:
            node_id: The ID of the node to explore around (e.g., "0042")
            collection_name: Name of the collection (to locate the source JSON)
            direction: Direction to explore - "up" (previous nodes), "down" (next nodes), or "both" (default: "both")
            count: Number of nodes to retrieve in each direction (default: 3, max: 10)
        
        Returns:
            JSON string containing the target node and surrounding nodes with full content
        """
        logger.info(f"[TOOL] explore_nodes called for node_id={node_id}, direction={direction}, count={count}")
        
        if deep_log_enabled:
            logger.info("="*80)
            logger.info("[DEEP LOG] explore_nodes INPUT:")
            logger.info(f"  node_id: {node_id}")
            logger.info(f"  collection_name: {collection_name}")
            logger.info(f"  direction: {direction}")
            logger.info(f"  count: {count}")
            logger.info("="*80)
        
        try:
            results = retrieval_tools_instance.explore_nodes(node_id, collection_name, direction, count)
            logger.info(f"[TOOL] explore_nodes returned context for node {node_id}")
            
            if deep_log_enabled:
                logger.info("="*80)
                logger.info("[DEEP LOG] explore_nodes OUTPUT:")
                if 'target_node' in results:
                    logger.info("\n  Target Node:")
                    logger.info(f"    node_id: {results['target_node'].get('node_id', 'N/A')}")
                    logger.info(f"    title: {results['target_node'].get('title', 'N/A')}")
                if 'nodes_above' in results:
                    logger.info(f"\n  Nodes Above: {len(results['nodes_above'])} nodes")
                    for node in results['nodes_above']:
                        logger.info(f"    - {node.get('node_id', 'N/A')}: {node.get('title', 'N/A')[:50]}...")
                if 'nodes_below' in results:
                    logger.info(f"\n  Nodes Below: {len(results['nodes_below'])} nodes")
                    for node in results['nodes_below']:
                        logger.info(f"    - {node.get('node_id', 'N/A')}: {node.get('title', 'N/A')[:50]}...")
                logger.info("="*80)
            
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] explore_nodes failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def list_collections() -> str:
        """
        List all available collections in the vector database.
        Use this to discover what collections are available for searching.
        
        Returns:
            JSON string containing collection names, counts, and metadata
        """
        logger.info("[TOOL] list_collections called")
        try:
            results = retrieval_tools_instance.list_collections()
            logger.info(f"[TOOL] list_collections returned {len(results)} collections")
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] list_collections failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def get_node(node_id: str, collection_name: str) -> str:
        """
        Fetch a single node by node_id from the collection's backing JSON structure.

        Returns:
            JSON string with the node (node_id/title/text/line_num/...) for inspection/citation.
        """
        logger.info(f"[TOOL] get_node called for node_id={node_id}")
        try:
            node = retrieval_tools_instance.get_node(node_id=node_id, collection_name=collection_name)
            return json.dumps({"node": node}, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] get_node failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def get_nodes(node_ids: List[str], collection_name: str) -> str:
        """
        Fetch multiple nodes by node_id (batch).

        Returns:
            JSON string with nodes in the same order as requested (missing IDs omitted).
        """
        logger.info(f"[TOOL] get_nodes called for {len(node_ids)} node_id(s)")
        try:
            nodes = retrieval_tools_instance.get_nodes(node_ids=node_ids, collection_name=collection_name)
            return json.dumps({"nodes": nodes, "count": len(nodes)}, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] get_nodes failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def expand_around(node_id: str, collection_name: str, radius: int = 3) -> str:
        """
        Expand around a node by radius above/below (document order).
        """
        logger.info(f"[TOOL] expand_around called for node_id={node_id}, radius={radius}")
        try:
            results = retrieval_tools_instance.expand_around(node_id=node_id, collection_name=collection_name, radius=radius)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] expand_around failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @tool
    def expand_many(seed_node_ids: List[str], collection_name: str, radius: int = 3, max_nodes: int = 50) -> str:
        """
        Expand around multiple seed nodes, dedupe, and return a stitched evidence set.
        """
        logger.info(f"[TOOL] expand_many called for {len(seed_node_ids)} seed(s), radius={radius}, max_nodes={max_nodes}")
        try:
            results = retrieval_tools_instance.expand_many(
                seed_node_ids=seed_node_ids, collection_name=collection_name, radius=radius, max_nodes=max_nodes
            )
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"[TOOL] expand_many failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    return [
        search_by_title,
        search_by_text,
        explore_nodes,
        list_collections,
        get_node,
        get_nodes,
        expand_around,
        expand_many,
    ]
