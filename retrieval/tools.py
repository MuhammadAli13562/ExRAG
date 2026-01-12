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
            self._node_index_cache: Dict[str, Dict[str, int]] = {}
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

    def _get_node_index_map(self, collection_name: str) -> Dict[str, int]:
        """Build or return a cached node_id -> index map for a collection's JSON."""
        # Use resolved JSON path as stable key
        collection = self.chroma_client.get_collection(collection_name)
        json_path = collection.metadata.get("json_source")
        if not json_path:
            raise ValueError(f"No JSON source found in collection metadata for {collection_name}")
        json_path_obj = Path(json_path)
        if not json_path_obj.is_absolute():
            project_root = Path(__file__).parent.parent
            json_path_obj = project_root / json_path
        cache_key = str(json_path_obj)

        if cache_key in self._node_index_cache:
            return self._node_index_cache[cache_key]

        nodes = self._load_json_from_collection(collection_name)
        index_map: Dict[str, int] = {}
        for idx, node in enumerate(nodes):
            nid = node.get("node_id")
            if isinstance(nid, str) and nid:
                index_map[nid] = idx

        self._node_index_cache[cache_key] = index_map
        return index_map

    def get_node(self, node_id: str, collection_name: str) -> Dict[str, Any]:
        """Return a single node by node_id from the collection's backing JSON."""
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span("tool.get_node", input={"node_id": node_id, "collection_name": collection_name}) as span:
            nodes = self._load_json_from_collection(collection_name)
            index_map = self._get_node_index_map(collection_name)
            if node_id not in index_map:
                err = f"Node {node_id} not found in JSON structure"
                if span is not None:
                    span.update(output={"error": err})
                raise ValueError(err)
            node = nodes[index_map[node_id]]
            if span is not None:
                span.update(
                    output={
                        "found": True,
                        "node_id": node.get("node_id"),
                        "title": node.get("title"),
                        "line_num": node.get("line_num"),
                    }
                )
            return node

    def get_nodes(self, node_ids: List[str], collection_name: str) -> List[Dict[str, Any]]:
        """Batch fetch nodes by node_id (preserving input order; missing nodes omitted)."""
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span("tool.get_nodes", input={"node_ids": node_ids[:200], "collection_name": collection_name}) as span:
            nodes = self._load_json_from_collection(collection_name)
            index_map = self._get_node_index_map(collection_name)
            out: List[Dict[str, Any]] = []
            missing: List[str] = []
            for node_id in node_ids:
                idx = index_map.get(node_id)
                if idx is not None:
                    out.append(nodes[idx])
                else:
                    missing.append(node_id)
            if span is not None:
                span.update(
                    output={
                        "requested": len(node_ids),
                        "returned": len(out),
                        "missing_count": len(missing),
                        "missing_ids_preview": missing[:20],
                    }
                )
            return out

    def expand_around(
        self,
        node_id: str,
        collection_name: str,
        radius: int = DEFAULT_EXPLORATION_COUNT,
    ) -> Dict[str, Any]:
        """Expand around a node by radius above/below (alias on top of explore_nodes)."""
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.expand_around",
            input={"node_id": node_id, "collection_name": collection_name, "radius": radius},
        ) as span:
            result = self.explore_nodes(node_id=node_id, collection_name=collection_name, direction="both", count=radius)
            nodes_above = result.get("nodes_above") or []
            nodes_below = result.get("nodes_below") or []
            if span is not None:
                span.update(
                    output={
                        "target_node_id": (result.get("target_node") or {}).get("node_id"),
                        "nodes_above": len(nodes_above),
                        "nodes_below": len(nodes_below),
                        "node_ids_preview": [n.get("node_id") for n in (nodes_above + nodes_below)[:30]],
                    }
                )
            return result

    def expand_many(
        self,
        seed_node_ids: List[str],
        collection_name: str,
        radius: int = DEFAULT_EXPLORATION_COUNT,
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        """
        Expand around multiple seed nodes, dedupe, and return a stitched evidence set.

        Returns:
            {
              "seed_node_ids": [...],
              "radius": int,
              "nodes": [ {node_id,title,text,line_num,...}, ... ],
              "node_ids": [ ... ],
              "total_nodes": int
            }
        """
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if radius < 0 or radius > MAX_EXPLORATION_DEPTH:
            raise ValueError(f"radius must be between 0 and {MAX_EXPLORATION_DEPTH}")

        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.expand_many",
            input={
                "seed_node_ids": seed_node_ids[:200],
                "collection_name": collection_name,
                "radius": radius,
                "max_nodes": max_nodes,
            },
        ) as span:
            nodes = self._load_json_from_collection(collection_name)
            index_map = self._get_node_index_map(collection_name)

            indices = set()
            for seed in seed_node_ids:
                idx = index_map.get(seed)
                if idx is None:
                    continue
                lo = max(0, idx - radius)
                hi = min(len(nodes) - 1, idx + radius)
                for j in range(lo, hi + 1):
                    indices.add(j)

            # Sort in document order (important for coherent context)
            ordered = sorted(indices)
            ordered = ordered[:max_nodes]
            out_nodes = [nodes[i] for i in ordered]
            out_ids = [n.get("node_id") for n in out_nodes if n.get("node_id")]

            out = {
                "seed_node_ids": seed_node_ids,
                "radius": radius,
                "max_nodes": max_nodes,
                "nodes": out_nodes,
                "node_ids": out_ids,
                "total_nodes": len(out_nodes),
            }
            if span is not None:
                span.update(output={"total_nodes": len(out_nodes), "node_ids_preview": out_ids[:50]})
            return out

    def explore_titles(
        self,
        node_id: str,
        collection_name: str,
        radius: int = 10,
        direction: Literal["up", "down", "both"] = "both"
    ) -> List[Dict[str, str]]:
        """
        Get titles of nodes around a target node (cheap exploration - no full text).

        This is used for structural navigation to quickly scan document structure
        without fetching full content.

        Args:
            node_id: Center node to explore around
            collection_name: Name of the collection
            radius: How many nodes in each direction (default: 10)
            direction: "up" (earlier in doc), "down" (later), or "both"

        Returns:
            List of {node_id, title, position} sorted by document order
            position: "above", "target", "below"
        """
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.explore_titles",
            input={"node_id": node_id, "collection_name": collection_name, "radius": radius, "direction": direction},
        ) as span:
            logger.info(f"Exploring titles: node_id={node_id}, radius={radius}, direction={direction}")
            try:
                # Load JSON and get node positions
                nodes = self._load_json_from_collection(collection_name)
                index_map = self._get_node_index_map(collection_name)

                if node_id not in index_map:
                    raise ValueError(f"Node {node_id} not found")

                target_idx = index_map[node_id]
                titles: List[Dict[str, str]] = []

                # Get nodes above
                if direction in ("up", "both"):
                    start_idx = max(0, target_idx - radius)
                    for idx in range(start_idx, target_idx):
                        n = nodes[idx]
                        titles.append({
                            "node_id": n.get("node_id", ""),
                            "title": n.get("title", ""),
                            "position": "above"
                        })

                # Add target node
                target_node = nodes[target_idx]
                titles.append({
                    "node_id": target_node.get("node_id", ""),
                    "title": target_node.get("title", ""),
                    "position": "target"
                })

                # Get nodes below
                if direction in ("down", "both"):
                    end_idx = min(len(nodes), target_idx + radius + 1)
                    for idx in range(target_idx + 1, end_idx):
                        n = nodes[idx]
                        titles.append({
                            "node_id": n.get("node_id", ""),
                            "title": n.get("title", ""),
                            "position": "below"
                        })

                logger.info(f"Explore titles returned {len(titles)} entries")
                if span is not None:
                    span.update(
                        output={
                            "num_titles": len(titles),
                            "titles_preview": [t.get("title", "")[:50] for t in titles[:10]],
                        }
                    )
                return titles

            except Exception as e:
                logger.error(f"Explore titles failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                raise RuntimeError(f"Explore titles failed: {e}")

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
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.search_by_title",
            input={"query": query, "collection_name": collection_name, "top_k": top_k},
        ) as span:
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
                    include=["documents", "metadatas", "distances"],
                )

                # Format results
                chunks = []
                for i in range(len(results["ids"][0])):
                    similarity_score = 1 - results["distances"][0][i]
                    chunk = {
                        "node_id": results["ids"][0][i],
                        "title": results["documents"][0][i],  # Title is stored as document
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity_score,
                        "source": "title_search",
                    }
                    chunks.append(chunk)
                    logger.debug(f"  Result {i+1}: node_id={chunk['node_id']}, score={similarity_score:.3f}")

                logger.info(f"Title search returned {len(chunks)} results")
                if span is not None:
                    span.update(
                        output={
                            "num_results": len(chunks),
                            "node_ids": [c.get("node_id") for c in chunks[:50]],
                            "top_scores": [c.get("similarity_score") for c in chunks[:10]],
                        }
                    )
                return chunks

            except Exception as e:
                logger.error(f"Title search failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
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
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.search_by_text",
            input={"query": query, "collection_name": collection_name, "top_k": top_k},
        ) as span:
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
                    include=["documents", "metadatas", "distances"],
                )

                # Format results
                chunks = []
                for i in range(len(results["ids"][0])):
                    similarity_score = 1 - results["distances"][0][i]
                    chunk = {
                        "node_id": results["ids"][0][i],
                        "text": results["documents"][0][i],  # Text is stored as document
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity_score,
                        "source": "text_search",
                    }
                    chunks.append(chunk)
                    logger.debug(f"  Result {i+1}: node_id={chunk['node_id']}, score={similarity_score:.3f}")

                logger.info(f"Text search returned {len(chunks)} results")
                if span is not None:
                    span.update(
                        output={
                            "num_results": len(chunks),
                            "node_ids": [c.get("node_id") for c in chunks[:50]],
                            "top_scores": [c.get("similarity_score") for c in chunks[:10]],
                        }
                    )
                return chunks

            except Exception as e:
                logger.error(f"Text search failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
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
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span(
            "tool.explore_nodes",
            input={"node_id": node_id, "collection_name": collection_name, "direction": direction, "count": count},
        ) as span:
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
                    "nodes_below": [],
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
                    "can_go_down": target_idx < len(nodes) - 1,
                }

                logger.info(f"Exploration complete: {len(result['nodes_above'])} above, {len(result['nodes_below'])} below")
                if span is not None:
                    span.update(
                        output={
                            "target_node_id": (result.get("target_node") or {}).get("node_id"),
                            "nodes_above": len(result.get("nodes_above") or []),
                            "nodes_below": len(result.get("nodes_below") or []),
                            "above_ids_preview": [n.get("node_id") for n in (result.get("nodes_above") or [])[:20]],
                            "below_ids_preview": [n.get("node_id") for n in (result.get("nodes_below") or [])[:20]],
                        }
                    )
                return result

            except Exception as e:
                logger.error(f"Node exploration failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                raise RuntimeError(f"Node exploration failed: {e}")
    
    def get_chapter_info(
        self,
        chapter_num: int,
        collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get chapter boundaries and section list.
        
        This enables deterministic structural navigation without LLM loops.
        (Anthropic best practice: clear Agent-Computer Interface)
        
        Args:
            chapter_num: Chapter number to look up
            collection_name: Name of the collection
            
        Returns:
            {
                "chapter": 4,
                "title": "Chapter 4: Cell Division",
                "start_idx": 100,
                "end_idx": 150,
                "start_node": "0100",
                "end_node": "0150",
                "num_nodes": 51,
                "sections": [
                    {"node_id": "0101", "title": "4.1 Introduction", "level": 2},
                    ...
                ]
            }
            or None if chapter not found
        """
        import re
        from .langfuse_tracing import get_tracer
        
        tracer = get_tracer()
        with tracer.span(
            "tool.get_chapter_info",
            input={"chapter_num": chapter_num, "collection_name": collection_name},
        ) as span:
            try:
                nodes = self._load_json_from_collection(collection_name)
                
                # Find chapter start
                chapter_start_idx = None
                chapter_title = None
                
                # Patterns to match chapter headers
                chapter_patterns = [
                    rf"^chapter\s+{chapter_num}\b",      # "Chapter 4" or "Chapter 4:"
                    rf"^ch\.?\s*{chapter_num}\b",        # "Ch. 4" or "Ch 4"
                    rf"^{chapter_num}\.\s",              # "4. Title"
                ]
                
                for idx, node in enumerate(nodes):
                    title = (node.get("title") or "").strip()
                    title_lower = title.lower()
                    
                    for pattern in chapter_patterns:
                        if re.match(pattern, title_lower):
                            chapter_start_idx = idx
                            chapter_title = title
                            break
                    
                    if chapter_start_idx is not None:
                        break
                
                if chapter_start_idx is None:
                    logger.info(f"[TOOLS] get_chapter_info: Chapter {chapter_num} not found")
                    if span is not None:
                        span.update(output={"found": False})
                    return None
                
                # Find next chapter start (to determine end boundary)
                chapter_end_idx = len(nodes)
                next_chapter = chapter_num + 1
                
                next_chapter_patterns = [
                    rf"^chapter\s+{next_chapter}\b",
                    rf"^ch\.?\s*{next_chapter}\b",
                    rf"^{next_chapter}\.\s",
                ]
                
                for idx in range(chapter_start_idx + 1, len(nodes)):
                    title = (nodes[idx].get("title") or "").strip()
                    title_lower = title.lower()
                    
                    for pattern in next_chapter_patterns:
                        if re.match(pattern, title_lower):
                            chapter_end_idx = idx
                            break
                    
                    if chapter_end_idx != len(nodes):
                        break
                
                # Extract sections within chapter
                sections = []
                for idx in range(chapter_start_idx, chapter_end_idx):
                    node = nodes[idx]
                    node_title = node.get("title") or ""
                    if node_title.strip():  # Only include nodes with titles
                        sections.append({
                            "node_id": node.get("node_id"),
                            "title": node_title,
                            "level": node.get("level", 1),
                            "idx": idx,
                        })
                
                result = {
                    "chapter": chapter_num,
                    "title": chapter_title,
                    "start_idx": chapter_start_idx,
                    "end_idx": chapter_end_idx,
                    "start_node": nodes[chapter_start_idx].get("node_id"),
                    "end_node": nodes[chapter_end_idx - 1].get("node_id") if chapter_end_idx > 0 else None,
                    "num_nodes": chapter_end_idx - chapter_start_idx,
                    "sections": sections,
                }
                
                logger.info(f"[TOOLS] get_chapter_info: Found chapter {chapter_num} "
                           f"(nodes {chapter_start_idx}-{chapter_end_idx}, {len(sections)} sections)")
                
                if span is not None:
                    span.update(output={
                        "found": True,
                        "start_idx": chapter_start_idx,
                        "end_idx": chapter_end_idx,
                        "num_sections": len(sections),
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"[TOOLS] get_chapter_info failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                return None

    def navigate_structural_deterministic(
        self,
        collection_name: str,
        chapter: Optional[int] = None,
        position: Optional[str] = None,  # "beginning" | "end" | None
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic structural navigation - no LLM loop needed.
        
        This implements Anthropic's "prompt chaining" principle:
        break tasks into clear, sequential steps.
        
        Steps:
        1. Find chapter landmark (if specified)
        2. Get chapter boundaries
        3. Filter by position (beginning/end)
        4. Match keywords within filtered range
        
        Args:
            collection_name: Name of the collection
            chapter: Chapter number to search within (optional)
            position: "beginning", "end", or None for full chapter
            keywords: Keywords to match in titles/text
            
        Returns:
            List of matching nodes with structural metadata
        """
        from .langfuse_tracing import get_tracer
        
        tracer = get_tracer()
        with tracer.span(
            "tool.navigate_structural_deterministic",
            input={"chapter": chapter, "position": position, "keywords": keywords},
        ) as span:
            logger.info(f"[TOOLS] Deterministic nav: chapter={chapter}, position={position}, keywords={keywords}")
            
            try:
                nodes = self._load_json_from_collection(collection_name)
                
                # Step 1 & 2: Get chapter boundaries
                if chapter is not None:
                    chapter_info = self.get_chapter_info(chapter, collection_name)
                    if chapter_info is None:
                        logger.warning(f"[TOOLS] Chapter {chapter} not found")
                        if span is not None:
                            span.update(output={"found": 0, "reason": "chapter_not_found"})
                        return []
                    
                    start_idx = chapter_info["start_idx"]
                    end_idx = chapter_info["end_idx"]
                    chapter_nodes = nodes[start_idx:end_idx]
                else:
                    # No chapter specified - search entire document
                    chapter_nodes = nodes
                    start_idx = 0
                    end_idx = len(nodes)
                
                # Step 3: Filter by position
                if position == "end":
                    # Last 25% of chapter or last 30 nodes, whichever is smaller
                    cutoff = max(len(chapter_nodes) - 30, int(len(chapter_nodes) * 0.75))
                    search_range = chapter_nodes[cutoff:]
                    position_offset = cutoff
                elif position == "beginning":
                    # First 25% of chapter or first 30 nodes, whichever is smaller
                    cutoff = min(30, int(len(chapter_nodes) * 0.25))
                    search_range = chapter_nodes[:cutoff]
                    position_offset = 0
                else:
                    search_range = chapter_nodes
                    position_offset = 0
                
                logger.info(f"[TOOLS] Search range: {len(search_range)} nodes "
                           f"(position={position}, offset={position_offset})")
                
                # Step 4: Match keywords
                matches = []
                keywords_lower = [kw.lower() for kw in (keywords or [])]
                
                for node in search_range:
                    title = (node.get("title") or "").lower()
                    text = (node.get("text") or "").lower()
                    
                    # If no keywords, return all nodes in range (for pure position queries)
                    if not keywords_lower:
                        matches.append({
                            **node,
                            "source": "structural",
                            "relevance_grade": "high",
                            "match_type": "position_only",
                        })
                        continue
                    
                    # Check keyword match
                    matched_keywords = []
                    for kw in keywords_lower:
                        if kw in title:
                            matched_keywords.append(kw)
                        elif kw in text[:500]:  # Check first 500 chars of text
                            matched_keywords.append(kw)
                    
                    if matched_keywords:
                        # Score based on match quality
                        title_match = any(kw in title for kw in matched_keywords)
                        grade = "high" if title_match else "medium"
                        
                        matches.append({
                            **node,
                            "source": "structural",
                            "relevance_grade": grade,
                            "match_type": "keyword",
                            "matched_keywords": matched_keywords,
                        })
                
                # Sort: title matches first, then by document order
                matches.sort(key=lambda x: (
                    0 if x.get("relevance_grade") == "high" else 1,
                    nodes.index(x) if x in nodes else 999999,
                ))
                
                # Limit results
                matches = matches[:10]
                
                logger.info(f"[TOOLS] Deterministic nav found {len(matches)} matches")
                
                if span is not None:
                    span.update(output={
                        "found": len(matches),
                        "node_ids": [m.get("node_id") for m in matches],
                        "grades": [m.get("relevance_grade") for m in matches],
                    })
                
                return matches
                
            except Exception as e:
                logger.error(f"[TOOLS] navigate_structural_deterministic failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                return []

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections with their metadata.
        
        Returns:
            List of collection information dictionaries
        """
        from .langfuse_tracing import get_tracer

        tracer = get_tracer()
        with tracer.span("tool.list_collections", input={}) as span:
            logger.info("Listing all collections")
            try:
                collections = self.chroma_client.list_collections()
                logger.debug(f"Found {len(collections)} collections")

                collection_info = []
                for coll in collections:
                    info = {
                        "name": coll.name,
                        "count": coll.count(),
                        "metadata": coll.metadata,
                    }
                    collection_info.append(info)
                    logger.debug(f"  Collection: {coll.name}, count={info['count']}")

                logger.info(f"Retrieved info for {len(collection_info)} collections")
                if span is not None:
                    span.update(
                        output={
                            "num_collections": len(collection_info),
                            "names": [c.get("name") for c in collection_info[:100]],
                        }
                    )
                return collection_info

            except Exception as e:
                logger.error(f"Failed to list collections: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                raise RuntimeError(f"Failed to list collections: {e}")

