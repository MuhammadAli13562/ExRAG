"""
Unified retrieval and agent pipeline interface.
Clean functions for querying, evaluation, and collection management.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.agent import query_agent as _query_agent_impl
from retrieval.tools import RetrievalTools
from common.models import AgentResult, EvalMetrics, RetrievalConfig
from common.settings import get_settings
from common.logging_config import get_logger


logger = get_logger("exrag.retrieval")


def query_agent(
    query: str,
    title_collection: str,
    text_collection: str,
    config: Optional[RetrievalConfig] = None,
    verbose: bool = False,
    deep_log: bool = False,
) -> AgentResult:
    """
    Query the retrieval agent with a single question.
    
    This is the main entry point for retrieval. It:
    - Initializes the LangGraph agent
    - Runs semantic search across title and text indexes
    - Uses explore_nodes for context expansion
    - Returns a cited answer
    
    Args:
        query: User question/query
        title_collection: Name of title-indexed collection
        text_collection: Name of text-indexed collection
        config: Retrieval configuration (uses defaults if None)
        verbose: If True, print intermediate steps
        deep_log: If True, log all tool inputs and outputs in detail
    
    Returns:
        AgentResult with answer and metadata
    
    Raises:
        ValueError: If collections don't exist or API key not configured
    """
    settings = get_settings()
    
    if config is None:
        config = RetrievalConfig(
            agent_model=settings.agent_model,
            temperature=settings.agent_temperature,
            top_k=settings.retrieval_top_k,
            similarity_threshold=settings.retrieval_similarity_threshold,
            max_iterations=settings.max_agent_iterations,
            recursion_limit=settings.agent_recursion_limit,
            exploration_count=settings.default_exploration_count,
        )
    
    # Check API key
    if not settings.validate_openai_key():
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    
    logger.info(f"Querying agent: '{query[:50]}...'")
    
    try:
        # Call the underlying agent implementation
        result = _query_agent_impl(
            user_query=query,
            title_collection=title_collection,
            text_collection=text_collection,
            verbose=verbose,
            deep_log=deep_log,
        )
        
        return AgentResult(
            query=query,
            answer=result["answer"],
            evidence_count=result["evidence_count"],
            title_collection=title_collection,
            text_collection=text_collection,
            success=True,
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return AgentResult(
            query=query,
            answer="",
            evidence_count=0,
            title_collection=title_collection,
            text_collection=text_collection,
            success=False,
            error=str(e),
        )


def evaluate_index(
    queries: List[str],
    title_collection: str,
    text_collection: str,
    chroma_dir: Optional[Path] = None,
    config: Optional[RetrievalConfig] = None,
    max_queries: Optional[int] = None,
    save_results: Optional[Path] = None,
) -> EvalMetrics:
    """
    Evaluate retrieval quality on a set of queries.
    
    Args:
        queries: List of query strings to evaluate
        title_collection: Name of title-indexed collection
        text_collection: Name of text-indexed collection
        chroma_dir: ChromaDB directory (uses settings default if None)
        config: Retrieval configuration (uses defaults if None)
        max_queries: Maximum number of queries to run (None = all)
        save_results: Optional path to save detailed results as JSONL
    
    Returns:
        EvalMetrics with aggregate statistics
    """
    settings = get_settings()
    
    if chroma_dir is None:
        chroma_dir = settings.chroma_db_dir
    
    if config is None:
        config = RetrievalConfig()
    
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
    
    logger.info(f"Evaluating {len(queries)} queries")
    
    results = []
    successful = 0
    failed = 0
    total_evidence = 0
    total_response_length = 0
    
    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}/{len(queries)}: {query[:50]}...")
        
        result = query_agent(
            query=query,
            title_collection=title_collection,
            text_collection=text_collection,
            config=config,
            verbose=False,
        )
        
        results.append(result)

        if result.success:
            successful += 1
            total_evidence += result.evidence_count
            total_response_length += len(result.answer)
        else:
            failed += 1

    # Calculate averages
    avg_evidence = total_evidence / successful if successful > 0 else 0
    avg_response_length = total_response_length / successful if successful > 0 else 0
    
    # Save detailed results if requested
    if save_results:
        save_results = Path(save_results)
        save_results.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_results, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result.model_dump_json() + '\n')
        
        logger.info(f"Detailed results saved to {save_results}")
    
    metrics = EvalMetrics(
        total_queries=len(queries),
        successful=successful,
        failed=failed,
        avg_evidence_count=avg_evidence,
        avg_response_length=avg_response_length,
        chroma_dir=chroma_dir,
        config=config,
    )
    
    logger.info(f"Evaluation complete: {successful}/{len(queries)} successful ({metrics.success_rate:.1%})")
    
    return metrics


def load_queries_from_file(queries_path: Path) -> List[str]:
    """
    Load queries from a text file (one per line) or JSONL.
    
    Args:
        queries_path: Path to queries file
    
    Returns:
        List of query strings
    """
    queries_path = Path(queries_path)
    
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    queries = []
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse as JSON (JSONL format)
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    query = data.get('query', data.get('question', ''))
                    if query:
                        queries.append(query)
                except json.JSONDecodeError:
                    queries.append(line)
            else:
                queries.append(line)
    
    logger.info(f"Loaded {len(queries)} queries from {queries_path.name}")
    return queries


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
    
    # Use the existing tools
    tools = RetrievalTools(persist_dir=chroma_dir)
    return tools.list_collections()

