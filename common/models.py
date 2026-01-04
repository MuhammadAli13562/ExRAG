"""
Pydantic models for pipeline data structures and results.
These models provide type safety, validation, and JSON serialization.
"""
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ===== Configuration Models =====

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    provider: str = Field(default="openai", description="Embedding provider")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for embedding")
    max_retries: int = Field(default=3, ge=1, description="Max retry attempts")
    retry_delay: int = Field(default=2, ge=1, description="Delay between retries (seconds)")
    index_field: Literal["text", "title", "summary", "prefix_summary"] = Field(
        default="text",
        description="Which field to index"
    )
    device: Optional[str] = Field(default=None, description="Device for local models (cpu/cuda)")
    
    class Config:
        frozen = False


class RetrievalConfig(BaseModel):
    """Configuration for retrieval and agent."""
    agent_model: str = Field(default="gpt-4o", description="LLM model for agent")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for LLM")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity")
    max_iterations: int = Field(default=15, ge=1, description="Max agent iterations")
    recursion_limit: int = Field(default=25, ge=1, description="LangGraph recursion limit")
    exploration_count: int = Field(default=3, ge=1, le=10, description="Nodes to explore")
    timeout: Optional[int] = Field(default=None, description="Query timeout (seconds)")
    
    class Config:
        frozen = False


# ===== Result Models =====

class TreeBuildResult(BaseModel):
    """Result from building a markdown tree."""
    doc_name: str = Field(description="Document name (without extension)")
    output_path: Path = Field(description="Path to output JSON file")
    total_nodes: int = Field(ge=0, description="Total nodes extracted")
    root_nodes: int = Field(ge=0, description="Number of root-level nodes")
    max_depth: int = Field(ge=0, description="Maximum tree depth")
    content_hash: str = Field(description="Hash of input content for caching")
    mode: Literal["simple", "full"] = Field(default="full", description="Build mode")
    timestamp: datetime = Field(default_factory=datetime.now, description="Build timestamp")
    skipped: bool = Field(default=False, description="True if build was skipped (cached)")
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to handle Path serialization."""
        data = self.model_dump(mode='json', **kwargs)
        data['output_path'] = str(data['output_path'])
        import json
        return json.dumps(data, default=str)


class IndexStats(BaseModel):
    """Statistics for a single index build."""
    collection_name: str = Field(description="ChromaDB collection name")
    index_field: str = Field(description="Field that was indexed")
    total_nodes: int = Field(ge=0, description="Total nodes in input")
    indexed: int = Field(ge=0, description="Successfully indexed nodes")
    skipped: int = Field(ge=0, description="Skipped nodes (empty/invalid)")
    failed: int = Field(ge=0, description="Failed nodes")
    batches: int = Field(ge=0, description="Number of batches processed")
    model: str = Field(description="Embedding model used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Index timestamp")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_nodes == 0:
            return 0.0
        return self.indexed / self.total_nodes


class EmbeddingRunResult(BaseModel):
    """Result from running the full embedding pipeline."""
    tree_json: Path = Field(description="Source tree JSON path")
    chroma_dir: Path = Field(description="ChromaDB directory")
    title_stats: Optional[IndexStats] = Field(default=None, description="Title index stats")
    text_stats: Optional[IndexStats] = Field(default=None, description="Text index stats")
    total_indexed: int = Field(ge=0, description="Total vectors indexed across all indexes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Run timestamp")
    config: EmbeddingConfig = Field(description="Configuration used")
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to handle Path serialization."""
        data = self.model_dump(mode='json', **kwargs)
        data['tree_json'] = str(data['tree_json'])
        data['chroma_dir'] = str(data['chroma_dir'])
        import json
        return json.dumps(data, default=str)


class AgentResult(BaseModel):
    """Result from a single agent query."""
    query: str = Field(description="User query")
    answer: str = Field(description="Agent answer")
    evidence_count: int = Field(ge=0, description="Number of evidence nodes collected")
    title_collection: str = Field(description="Title collection used")
    text_collection: str = Field(description="Text collection used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")
    success: bool = Field(default=True, description="Whether query succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EvalMetrics(BaseModel):
    """Evaluation metrics for retrieval quality."""
    total_queries: int = Field(ge=0, description="Total queries evaluated")
    successful: int = Field(ge=0, description="Successful queries")
    failed: int = Field(ge=0, description="Failed queries")
    avg_evidence_count: float = Field(ge=0.0, description="Average evidence nodes per query")
    avg_response_length: float = Field(ge=0.0, description="Average response length (chars)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Eval timestamp")
    queries_path: Optional[Path] = Field(default=None, description="Source queries file")
    chroma_dir: Path = Field(description="ChromaDB directory evaluated")
    config: RetrievalConfig = Field(description="Configuration used")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_queries == 0:
            return 0.0
        return self.successful / self.total_queries
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to handle Path serialization."""
        data = self.model_dump(mode='json', **kwargs)
        data['chroma_dir'] = str(data['chroma_dir'])
        if data.get('queries_path'):
            data['queries_path'] = str(data['queries_path'])
        import json
        return json.dumps(data, default=str)


# ===== Node Models (for internal use) =====

class Node(BaseModel):
    """Represents a single node from the tree structure."""
    node_id: str = Field(description="Unique node ID")
    title: str = Field(description="Node title")
    text: str = Field(description="Node text content")
    line_num: Optional[int] = Field(default=None, description="Line number in source")
    level: Optional[int] = Field(default=None, description="Heading level")
    
    class Config:
        extra = "allow"  # Allow additional fields from JSON


# ===== Summary Model for CLI output =====

class PipelineSummary(BaseModel):
    """High-level summary of pipeline run for CLI output."""
    stage: str = Field(description="Pipeline stage: tree, embedding, or retrieval")
    status: Literal["success", "failure", "skipped"] = Field(description="Overall status")
    message: str = Field(description="Human-readable summary message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Summary timestamp")

