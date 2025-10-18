# PageIndex Pipeline Guide

## Overview

PageIndex is a hierarchical document indexing and agentic retrieval pipeline with three main stages:

```
Markdown Files → Tree Structure → Embedding Indexes → Agentic Retrieval
     (md/)      →   (results/)   →  (embeddings/)    →   (queries)
```

## Architecture

### 1. **Tree Building** (`md_to_tree.py`)
- **Input**: Markdown files with hierarchical headers
- **Output**: JSON tree structure with node IDs, titles, and text
- **Features**: Content hashing for caching, automatic node ID assignment

### 2. **Embedding Pipeline** (`embeddings/`)
- **Input**: Tree JSON files
- **Output**: ChromaDB collections (title and/or text indexes)
- **Features**: Batch embedding, retry logic, collection management

### 3. **Retrieval Agent** (`retrieval/`)
- **Input**: User queries + collection names
- **Output**: Cited answers from LangGraph agent
- **Features**: Multi-tool search (title/text), context exploration, citation tracking

### 4. **Common** (`common/`)
- Shared Pydantic models for type safety
- Centralized settings with `.env` support
- Structured logging

## Entry Points

### Option 1: Per-Package CLIs (Recommended for modularity)

```bash
# Build tree
python md_to_tree.py build md/biology.md

# Build embeddings
python -m embeddings build results/biology_structure.json

# Query
python -m retrieval query "What are electric charges?" \
  --title biology_titles --text biology_texts
```

### Option 2: Unified CLI (Recommended for convenience)

```bash
# Individual stages
python pageindex_cli.py tree md/biology.md
python pageindex_cli.py embed results/biology_structure.json
python pageindex_cli.py query "..." --title X --text Y

# End-to-end pipeline
python pageindex_cli.py pipeline md/biology.md
```

### Option 3: Python API (Recommended for Prefect/orchestration)

```python
from md_to_tree import build_tree
from embeddings.pipeline import build_embedding_indexes
from retrieval.pipeline import query_agent
from common.models import EmbeddingConfig, RetrievalConfig

# Build tree
tree_result = build_tree(md_path="md/biology.md")

# Build embeddings
embed_result = build_embedding_indexes(
    tree_json=tree_result.output_path,
    which="both",
)

# Query
agent_result = query_agent(
    query="What are electric charges?",
    title_collection=embed_result.title_stats.collection_name,
    text_collection=embed_result.text_stats.collection_name,
)

print(agent_result.answer)
```

## Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Settings Hierarchy

1. **Defaults**: `common/settings.py`
2. **Environment**: `.env` file (prefix: `PAGEINDEX_`)
3. **CLI flags**: Override per-command

Example:
```bash
# Use default model
python -m embeddings build results/biology.json

# Override model via CLI
python -m embeddings build results/biology.json --model text-embedding-3-large

# Override model via .env
# In .env: PAGEINDEX_EMBEDDING_MODEL=text-embedding-3-large
python -m embeddings build results/biology.json
```

## Pipeline Functions

### Tree Building

**Function**: `build_tree(md_path, output_path, mode, force)`

**Returns**: `TreeBuildResult`
- `doc_name`: Document name
- `output_path`: Path to JSON
- `total_nodes`: Node count
- `content_hash`: For caching
- `skipped`: True if unchanged

**Behavior**:
- Parses markdown headers (h1-h6)
- Builds hierarchical tree
- Assigns sequential node IDs
- Caches based on content hash

### Embedding Building

**Function**: `build_embedding_indexes(tree_json, chroma_dir, config, which, collection_prefix, reset)`

**Returns**: `EmbeddingRunResult`
- `title_stats`: Title index statistics
- `text_stats`: Text index statistics
- `total_indexed`: Total vectors

**Behavior**:
- Loads nodes from tree JSON
- Creates ChromaDB collections
- Embeds in batches with retries
- Stores metadata (source JSON, model, field)

### Retrieval Query

**Function**: `query_agent(query, title_collection, text_collection, config, verbose)`

**Returns**: `AgentResult`
- `answer`: Cited answer
- `iteration_count`: Agent iterations
- `success`: True/False

**Behavior**:
- Initializes LangGraph agent
- Searches title + text indexes
- Explores context via `explore_nodes`
- Enforces citation requirement

### Evaluation

**Function**: `evaluate_index(queries, title_collection, text_collection, ...)`

**Returns**: `EvalMetrics`
- `success_rate`: % successful queries
- `avg_iteration_count`: Avg iterations
- `avg_response_length`: Avg chars

## CLI Reference

### Tree Building

```bash
# Build tree (auto-generates output path in results/)
python md_to_tree.py build md/biology.md

# Custom output path
python md_to_tree.py build md/physics.md --output custom/path.json

# Force rebuild (ignore cache)
python md_to_tree.py build md/biology.md --force

# JSON output (for scripting)
python md_to_tree.py build md/biology.md --json

# Print tree structure
python md_to_tree.py print results/biology_structure.json
```

### Embedding Building

```bash
# Build both indexes
python -m embeddings build results/biology_structure.json

# Build only title index
python -m embeddings build results/biology_structure.json --which titles

# Custom collection prefix
python -m embeddings build results/biology_structure.json --prefix bio_v2

# Reset existing collections
python -m embeddings build results/biology_structure.json --reset

# Custom model
python -m embeddings build results/biology_structure.json --model text-embedding-3-large

# List collections
python -m embeddings list

# Delete collection
python -m embeddings delete biology_titles

# Collection info
python -m embeddings info biology_titles
```

### Retrieval & Query

```bash
# Single query
python -m retrieval query "What are electric charges?" \
  --title biology_titles --text biology_texts

# Verbose output (shows agent steps)
python -m retrieval query "..." --title X --text Y --verbose

# Interactive mode
python -m retrieval interactive --title X --text Y

# Evaluate on query set
python -m retrieval eval queries.txt --title X --text Y

# Save evaluation results
python -m retrieval eval queries.txt --title X --text Y \
  --save results/eval.jsonl

# List collections
python -m retrieval list
```

### Unified CLI

```bash
# End-to-end pipeline
python pageindex_cli.py pipeline md/biology.md

# With options
python pageindex_cli.py pipeline md/biology.md \
  --reset-indexes --verbose

# Individual stages
python pageindex_cli.py tree md/biology.md
python pageindex_cli.py embed results/biology_structure.json
python pageindex_cli.py query "..." --title X --text Y

# List collections
python pageindex_cli.py list
```

## Integration with Prefect

### Example Flow

```python
from prefect import flow, task
from md_to_tree import build_tree
from embeddings.pipeline import build_embedding_indexes
from retrieval.pipeline import evaluate_index, load_queries_from_file
from common.models import EmbeddingConfig, RetrievalConfig

@task
def build_tree_task(md_path):
    return build_tree(md_path=md_path)

@task
def build_embeddings_task(tree_json):
    return build_embedding_indexes(
        tree_json=tree_json,
        which="both",
    )

@task
def evaluate_task(title_coll, text_coll, queries):
    return evaluate_index(
        queries=queries,
        title_collection=title_coll,
        text_collection=text_coll,
    )

@flow(name="pageindex-pipeline")
def pageindex_flow(md_path, queries_path):
    # Build tree
    tree_result = build_tree_task(md_path)
    
    # Build embeddings
    embed_result = build_embeddings_task(tree_result.output_path)
    
    # Evaluate
    queries = load_queries_from_file(queries_path)
    metrics = evaluate_task(
        embed_result.title_stats.collection_name,
        embed_result.text_stats.collection_name,
        queries,
    )
    
    return {
        "tree": tree_result,
        "embeddings": embed_result,
        "metrics": metrics,
    }

# Run flow
if __name__ == "__main__":
    result = pageindex_flow("md/biology.md", "queries.txt")
    print(f"Success rate: {result['metrics'].success_rate:.1%}")
```

### Change-Based Trigger

```python
import hashlib
from pathlib import Path
from prefect import flow, task

@task
def compute_md_hash(md_dir: Path):
    """Compute hash of all markdown files."""
    hasher = hashlib.sha256()
    for md_file in sorted(md_dir.glob("*.md")):
        hasher.update(md_file.read_bytes())
    return hasher.hexdigest()

@flow(name="pageindex-incremental")
def incremental_flow(md_dir: Path):
    current_hash = compute_md_hash(md_dir)
    
    # Load previous hash (from state or file)
    previous_hash = load_previous_hash()
    
    if current_hash == previous_hash:
        print("No changes detected, skipping pipeline")
        return
    
    # Run pipeline for changed files
    for md_file in md_dir.glob("*.md"):
        tree_result = build_tree_task(md_file)
        embed_result = build_embeddings_task(tree_result.output_path)
    
    # Save current hash
    save_hash(current_hash)
```

## Testing

```bash
# Test tree building
python md_to_tree.py build md/biology.md --verbose

# Test embeddings
python -m embeddings build results/biology_structure.json --verbose

# Test retrieval
python -m retrieval query "test query" \
  --title biology_titles --text biology_texts --verbose

# Full pipeline test
python pageindex_cli.py pipeline md/biology.md --verbose
```

## Troubleshooting

### "OPENAI_API_KEY not set"
- Create `.env` file from `.env.example`
- Set `OPENAI_API_KEY=your_key`

### "Collection not found"
- Run `python -m embeddings list` to see available collections
- Use exact collection names (case-sensitive)

### "No nodes found in JSON"
- Check tree JSON with `python md_to_tree.py print results/biology.json`
- Verify markdown has headers (# ## ### etc.)

### Embedding rate limits
- Adjust batch size: `--batch-size 50`
- Increase retry delay in settings

## Next Steps

1. **Install Prefect**: `pip install prefect`
2. **Create flows**: See examples above
3. **Schedule runs**: Use Prefect deployments
4. **Add sensors**: Watch `md/` directory for changes
5. **Add partitions**: Process files independently
6. **Add observability**: Log metrics to Prefect artifacts

