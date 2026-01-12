# ExRAG Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ExRAG Pipeline                          │
│                  Hierarchical Document Indexing &                │
│                      Agentic Retrieval System                    │
└─────────────────────────────────────────────────────────────────┘

┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ Markdown  │────▶│ Tree Builder │────▶│  Embeddings  │────▶│ Agent   │
│   Files   │     │              │     │   Pipeline   │     │ Queries │
└───────────┘     └──────────────┘     └──────────────┘     └─────────┘
    md/*.md        results/*.json       chroma_db/           Answers
```

## Component Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         common/                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   models.py  │  │ settings.py  │  │  logging_config.py   │  │
│  │              │  │              │  │                      │  │
│  │ • TreeBuild  │  │ • Settings   │  │ • setup_logging()   │  │
│  │   Result     │  │ • BaseSettings│  │ • get_logger()      │  │
│  │ • Embedding  │  │ • .env support│  │                      │  │
│  │   RunResult  │  │              │  │                      │  │
│  │ • AgentResult│  │              │  │                      │  │
│  │ • EvalMetrics│  │              │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    Used by all modules below
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────────┐
│md_to_tree.py│        │ embeddings/ │        │   retrieval/    │
├─────────────┤        ├─────────────┤        ├─────────────────┤
│             │        │             │        │                 │
│build_tree() │        │pipeline.py  │        │ pipeline.py     │
│             │        │ • build_    │        │  • query_agent()│
│Returns:     │        │   embedding_│        │  • evaluate_    │
│TreeBuild    │        │   indexes() │        │    index()      │
│Result       │        │             │        │                 │
│             │        │cli.py       │        │ cli.py          │
│CLI:         │        │ • build     │        │  • query        │
│ • build     │        │ • list      │        │  • interactive  │
│ • print     │        │ • delete    │        │  • eval         │
│             │        │ • info      │        │  • list         │
└─────────────┘        └─────────────┘        └─────────────────┘
```

## Data Flow

### 1. Tree Building Flow

```
┌───────────┐
│ Markdown  │
│   File    │
│ (*.md)    │
└─────┬─────┘
      │
      ▼
┌─────────────────────────┐
│ extract_nodes_from_     │
│ markdown()              │
│ • Parse headers         │
│ • Skip code blocks      │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ extract_node_text_      │
│ content()               │
│ • Extract content       │
│ • Assign line numbers   │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ build_tree_from_nodes() │
│ • Build hierarchy       │
│ • Assign node IDs       │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ clean_empty_nodes()     │
│ • Remove empty arrays   │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│  TreeBuildResult        │
│  ┌───────────────────┐  │
│  │ doc_name          │  │
│  │ output_path       │  │
│  │ total_nodes: 42   │  │
│  │ root_nodes: 3     │  │
│  │ max_depth: 4      │  │
│  │ content_hash      │  │
│  └───────────────────┘  │
└─────────────────────────┘
      │
      ▼
┌─────────────────┐
│ results/        │
│ biology_        │
│ structure.json  │
└─────────────────┘
```

### 2. Embedding Flow

```
┌─────────────────┐
│ Tree JSON       │
│ structure.json  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ load_nodes_from_tree()  │
│ • Load JSON             │
│ • Flatten hierarchy     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ VectorIndexer           │
│ .index_nodes()          │
│ • Batch processing      │
│ • Embed with OpenAI     │
│ • Retry logic           │
└────────┬────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌──────┐
│Title │  │ Text │
│Index │  │Index │
└──┬───┘  └───┬──┘
   │          │
   ▼          ▼
┌─────────────────────────┐
│  EmbeddingRunResult     │
│  ┌───────────────────┐  │
│  │ title_stats:      │  │
│  │   indexed: 42/42  │  │
│  │ text_stats:       │  │
│  │   indexed: 42/42  │  │
│  │ total_indexed: 84 │  │
│  └───────────────────┘  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ChromaDB Collections    │
│ ┌─────────────────────┐ │
│ │ bio_titles          │ │
│ │ (42 vectors)        │ │
│ ├─────────────────────┤ │
│ │ bio_texts           │ │
│ │ (42 vectors)        │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### 3. Retrieval Flow

```
┌───────────┐
│   User    │
│   Query   │
└─────┬─────┘
      │
      ▼
┌─────────────────────────┐
│ query_agent()           │
│ • Initialize LangGraph  │
│ • Create agent state    │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ LangGraph Agent         │
│ ┌─────────────────────┐ │
│ │ 1. planner()        │ │
│ │    ▼                │ │
│ │ 2. retrieve()       │ │
│ │    ▼                │ │
│ │ 3. expand()         │ │
│ │    ▼                │ │
│ │ 4. synthesize()     │ │
│ │    ▼                │ │
│ │ 5. validate()       │ │
│ └─────────────────────┘ │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Tool: search_by_text()  │
│ ┌─────────────────────┐ │
│ │ ChromaDB query      │ │
│ │ Collection: texts   │ │
│ │ Returns: node_ids   │ │
│ └─────────────────────┘ │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Tool: explore_nodes()   │
│ ┌─────────────────────┐ │
│ │ Load source JSON    │ │
│ │ Get adjacent nodes  │ │
│ │ Returns: context    │ │
│ └─────────────────────┘ │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│  AgentResult            │
│  ┌───────────────────┐  │
│  │ answer: "..."     │  │
│  │ citations: [Node  │  │
│  │   0005, 0006]     │  │
│  │ iteration_count: 3│  │
│  │ success: True     │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

## Configuration System

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Hierarchy                    │
└─────────────────────────────────────────────────────────────┘

Priority (highest to lowest):

1. CLI Flags
   └─▶ python -m embeddings build ... --model text-embedding-3-large

2. Environment Variables (.env file)
   └─▶ ExRAG_EMBEDDING_MODEL=text-embedding-3-large

3. Defaults (common/settings.py)
   └─▶ embedding_model: str = "text-embedding-3-small"

┌─────────────────────────────────────────────────────────────┐
│                      Settings Class                          │
├─────────────────────────────────────────────────────────────┤
│ class Settings(BaseSettings):                               │
│   # Paths                                                    │
│   project_root: Path                                         │
│   md_dir: Path                                               │
│   results_dir: Path                                          │
│   chroma_db_dir: Path                                        │
│                                                              │
│   # OpenAI                                                   │
│   openai_api_key: str                                        │
│                                                              │
│   # Embedding                                                │
│   embedding_model: str = "text-embedding-3-small"           │
│   embedding_batch_size: int = 100                           │
│                                                              │
│   # Agent                                                    │
│   agent_model: str = "gpt-4o"                               │
│   retrieval_top_k: int = 5                                  │
│   max_agent_iterations: int = 15                            │
│                                                              │
│   # Logging                                                  │
│   log_level: str = "INFO"                                   │
│   log_to_file: bool = True                                  │
└─────────────────────────────────────────────────────────────┘

Usage:
  from common.settings import get_settings
  settings = get_settings()  # Cached singleton
  print(settings.embedding_model)
```

## Entry Point Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Entry Points                            │
└─────────────────────────────────────────────────────────────┘

1. Per-Module CLIs (python -m package)
   ┌──────────────────────────────────────────────────┐
   │ python -m embeddings build ...                   │
   │   ▼                                              │
   │ embeddings/__main__.py                           │
   │   ▼                                              │
   │ embeddings/cli.py (Typer app)                    │
   │   ▼                                              │
   │ embeddings/pipeline.py (business logic)          │
   │   ▼                                              │
   │ Returns: EmbeddingRunResult                      │
   └──────────────────────────────────────────────────┘

2. Direct Module CLIs (python file.py)
   ┌──────────────────────────────────────────────────┐
   │ python md_to_tree.py build ...                   │
   │   ▼                                              │
   │ Typer app in __main__ block                      │
   │   ▼                                              │
   │ build_tree() function                            │
   │   ▼                                              │
   │ Returns: TreeBuildResult                         │
   └──────────────────────────────────────────────────┘

3. Unified CLI (convenience wrapper)
   ┌──────────────────────────────────────────────────┐
   │ python ExRAG_cli.py pipeline ...             │
   │   ▼                                              │
   │ ExRAG_cli.py (Typer app)                     │
   │   ▼                                              │
   │ Imports and calls:                               │
   │   • build_tree()                                 │
   │   • build_embedding_indexes()                    │
   │   • query_agent()                                │
   │   ▼                                              │
   │ Formats output                                   │
   └──────────────────────────────────────────────────┘

4. Python API (orchestration)
   ┌──────────────────────────────────────────────────┐
   │ from md_to_tree import build_tree                │
   │ from embeddings.pipeline import build_indexes    │
   │                                                  │
   │ result = build_tree("md/biology.md")             │
   │ # Direct function call, no subprocess            │
   │ # Returns typed Pydantic model                   │
   └──────────────────────────────────────────────────┘
```

## Orchestration Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefect Integration                       │
└─────────────────────────────────────────────────────────────┘

@flow
def ExRAG_flow(md_path):
    
    @task
    def build_tree_task(path):
        return build_tree(md_path=path)
    
    @task
    def build_embeddings_task(tree_json):
        return build_embedding_indexes(tree_json=tree_json)
    
    @task
    def evaluate_task(title, text, queries):
        return evaluate_index(
            queries=queries,
            title_collection=title,
            text_collection=text,
        )
    
    # Flow execution
    tree = build_tree_task(md_path)
    embed = build_embeddings_task(tree.output_path)
    metrics = evaluate_task(
        embed.title_stats.collection_name,
        embed.text_stats.collection_name,
        load_queries("queries.txt"),
    )
    
    return metrics


Benefits:
• No subprocess calls
• Typed results
• Automatic retries
• Caching support
• Dependency graph
• Observability
```

## Data Models

```
┌─────────────────────────────────────────────────────────────┐
│                   Pydantic Models (common/models.py)         │
└─────────────────────────────────────────────────────────────┘

TreeBuildResult                 EmbeddingRunResult
┌─────────────────────┐        ┌──────────────────────┐
│ doc_name: str       │        │ tree_json: Path      │
│ output_path: Path   │        │ chroma_dir: Path     │
│ total_nodes: int    │        │ title_stats: IndexStats
│ root_nodes: int     │        │ text_stats: IndexStats
│ max_depth: int      │        │ total_indexed: int   │
│ content_hash: str   │        │ config: EmbeddingConfig
│ mode: "full"        │        │ timestamp: datetime  │
│ timestamp: datetime │        └──────────────────────┘
└─────────────────────┘

IndexStats                      AgentResult
┌─────────────────────┐        ┌──────────────────────┐
│ collection_name: str│        │ query: str           │
│ index_field: str    │        │ answer: str          │
│ total_nodes: int    │        │ iteration_count: int │
│ indexed: int        │        │ title_collection: str│
│ skipped: int        │        │ text_collection: str │
│ failed: int         │        │ success: bool        │
│ batches: int        │        │ error: Optional[str] │
│ model: str          │        │ timestamp: datetime  │
│ timestamp: datetime │        └──────────────────────┘
└─────────────────────┘

All models support:
• Validation
• JSON serialization (.model_dump_json())
• Dict conversion (.model_dump())
• Type hints for IDE
```

## Logging Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Logging Flow                           │
└─────────────────────────────────────────────────────────────┘

from common.logging_config import get_logger

logger = get_logger("ExRAG.embeddings")
                    │
                    ▼
            ┌───────────────┐
            │ Logger Config │
            ├───────────────┤
            │ Name: ExRAG.embeddings
            │ Level: INFO (from settings)
            │ Handlers:
            │   • Console (stdout)
            │   • File (ExRAG.log)
            └───────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼                               ▼
┌─────────┐                   ┌──────────┐
│ Console │                   │   File   │
│ Output  │                   │ ExRAG│
│ (INFO+) │                   │ .log     │
│         │                   │ (DEBUG+) │
└─────────┘                   └──────────┘

Format:
2025-10-18 10:30:45 - ExRAG.embeddings - INFO - Indexed 42 nodes
```

## Summary

### Key Architectural Principles

1. **Separation of Concerns**
   - Business logic in pure functions
   - CLI parsing in Typer apps
   - Configuration in centralized settings

2. **Type Safety**
   - Pydantic models throughout
   - Validation at boundaries
   - IDE support

3. **Composability**
   - Small, focused functions
   - Clean imports
   - No circular dependencies

4. **Observability**
   - Structured logging
   - Detailed metrics
   - Progress tracking

5. **Orchestration-Ready**
   - No subprocess calls needed
   - Typed results
   - Idempotent operations

6. **Configuration Hierarchy**
   - Defaults → Environment → CLI
   - Single source of truth
   - Easy overrides

