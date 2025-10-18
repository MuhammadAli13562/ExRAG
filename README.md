# ExRAG

**Hierarchical document indexing and agentic retrieval pipeline**

ExRAG converts markdown documents into searchable vector indexes with hierarchical structure preservation, enabling intelligent retrieval through a LangGraph-powered agent.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 3. Run the pipeline
python exrag_cli.py pipeline md/biology.md

# 4. Query the index
python exrag_cli.py query "What are electric charges?" \
  --title biology_structure__simple_12345678_titles \
  --text biology_structure__simple_12345678_texts
```

## 📋 Features

- **🌳 Hierarchical Tree Building**: Parse markdown headers into structured JSON with node IDs
- **🔍 Dual Indexing**: Separate title and text embeddings for optimized retrieval
- **🤖 Agentic Retrieval**: LangGraph agent with multi-tool search and context exploration
- **📊 Type-Safe Pipeline**: Pydantic models throughout for validation and serialization
- **⚙️ Flexible Configuration**: Environment variables, CLI flags, or Python API
- **🔄 Orchestration-Ready**: Clean function APIs designed for Prefect/Dagster integration
- **💾 Caching**: Content-based hashing to skip unchanged documents
- **📝 Structured Logging**: Consistent logging across all components

## 📁 Project Structure

```
ExRAG/
├── common/                    # Shared models, settings, logging
│   ├── __init__.py
│   ├── models.py             # Pydantic models (TreeBuildResult, etc.)
│   ├── settings.py           # Centralized configuration
│   └── logging_config.py     # Logging setup
│
├── tree/                     # Tree building module
│   ├── __init__.py
│   ├── __main__.py          # python -m tree
│   ├── cli.py               # Typer CLI
│   ├── pipeline.py          # Main API: build_tree()
│   └── builder.py           # Core building logic
│
├── embeddings/               # Embedding pipeline
│   ├── __init__.py
│   ├── __main__.py          # python -m embeddings
│   ├── cli.py               # Typer CLI
│   ├── pipeline.py          # Main API: build_embedding_indexes()
│   ├── indexer.py           # ChromaDB indexing logic
│   ├── embedder.py          # OpenAI embedding wrapper
│   └── node_loader.py       # JSON tree loader
│
├── retrieval/               # Retrieval & agent
│   ├── __init__.py
│   ├── __main__.py          # python -m retrieval
│   ├── cli.py               # Typer CLI
│   ├── pipeline.py          # Main API: query_agent(), evaluate_index()
│   ├── agent.py             # LangGraph agent implementation
│   ├── tools.py             # Agent tools (search, explore)
│   └── config.py            # Legacy config (migrate to common/)
│
├── exrag_cli.py             # Unified CLI wrapper
├── .env.example             # Configuration template
├── PIPELINE.md              # Detailed pipeline guide
└── README.md                # This file
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repo-url>
cd PageIndex

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## 💻 Usage

### Option 1: Unified CLI (Recommended for quick start)

```bash
# Full pipeline (tree + embeddings)
python exrag_cli.py pipeline md/biology.md

# Individual stages
python exrag_cli.py tree md/physics.md
python exrag_cli.py embed results/physics_structure.json
python exrag_cli.py query "What is Newton's law?" --title X --text Y

# List collections
python exrag_cli.py list
```

### Option 2: Per-Module CLIs (Recommended for advanced use)

```bash
# Tree building
python -m tree build md/biology.md --verbose
python -m tree print results/biology_structure.json

# Embedding building
python -m embeddings build results/biology_structure.json
python -m embeddings build results/biology_structure.json --which titles --reset
python -m embeddings list
python -m embeddings info biology_titles

# Retrieval
python -m retrieval query "What are electric charges?" --title X --text Y
python -m retrieval interactive --title X --text Y
python -m retrieval eval queries.txt --title X --text Y --save results/eval.jsonl
python -m retrieval list
```

### Option 3: Python API (Recommended for orchestration)

```python
from pathlib import Path
from tree.pipeline import build_tree
from embeddings.pipeline import build_embedding_indexes
from retrieval.pipeline import query_agent
from common.models import EmbeddingConfig, RetrievalConfig

# 1. Build tree
tree_result = build_tree(
    md_path=Path("md/biology.md"),
    force=False,  # Use cache if unchanged
)

print(f"Built tree: {tree_result.total_nodes} nodes")

# 2. Build embeddings
embed_result = build_embedding_indexes(
    tree_json=tree_result.output_path,
    which="both",  # Build title and text indexes
    reset=False,
)

print(f"Indexed {embed_result.total_indexed} vectors")

# 3. Query
agent_result = query_agent(
    query="What are the two kinds of electric charges?",
    title_collection=embed_result.title_stats.collection_name,
    text_collection=embed_result.text_stats.collection_name,
    verbose=True,
)

print(f"\nAnswer:\n{agent_result.answer}")
print(f"Iterations: {agent_result.iteration_count}")
```

## 🔄 Pipeline Stages

### 1️⃣ Tree Building

**Input**: Markdown files with hierarchical headers  
**Output**: JSON tree structure  
**Function**: `build_tree(md_path, output_path, mode, force)`

```python
from tree.pipeline import build_tree

result = build_tree(
    md_path="md/biology.md",
    mode="full",  # or "simple"
    force=False,  # Skip if unchanged (content hash)
)

# Result contains:
# - doc_name: Document name
# - output_path: Path to JSON
# - total_nodes: Number of nodes
# - content_hash: For caching
# - skipped: True if cached
```

### 2️⃣ Embedding Building

**Input**: Tree JSON files  
**Output**: ChromaDB collections  
**Function**: `build_embedding_indexes(tree_json, config, which, ...)`

```python
from embeddings.pipeline import build_embedding_indexes
from common.models import EmbeddingConfig

result = build_embedding_indexes(
    tree_json="results/biology_structure.json",
    which="both",  # "titles", "texts", or "both"
    config=EmbeddingConfig(
        model="text-embedding-3-small",
        batch_size=100,
    ),
    reset=False,
)

# Result contains:
# - title_stats: Title index statistics
# - text_stats: Text index statistics
# - total_indexed: Total vectors
```

### 3️⃣ Retrieval & Query

**Input**: Query + collection names  
**Output**: Cited answer  
**Function**: `query_agent(query, title_collection, text_collection, ...)`

```python
from retrieval.pipeline import query_agent

result = query_agent(
    query="What are electric charges?",
    title_collection="biology_titles",
    text_collection="biology_texts",
    verbose=True,  # Show agent steps
)

# Result contains:
# - answer: Cited answer with [Node XXXX] references
# - iteration_count: Agent iterations
# - success: True/False
```

## ⚙️ Configuration

### Environment Variables

Create `.env` from template:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional overrides (with defaults shown)
EXRAG_EMBEDDING_MODEL=text-embedding-3-small
EXRAG_EMBEDDING_BATCH_SIZE=100
EXRAG_AGENT_MODEL=gpt-4o
EXRAG_RETRIEVAL_TOP_K=5
EXRAG_LOG_LEVEL=INFO
```

### Settings Hierarchy

1. **Defaults**: Defined in `common/settings.py`
2. **Environment**: `.env` file (prefix: `PAGEINDEX_`)
3. **CLI flags**: Override per-command
4. **Python API**: Pass config objects directly

## 🔌 Orchestration with Prefect

PageIndex is designed for easy integration with workflow orchestrators like Prefect:

```python
from prefect import flow, task
from tree.pipeline import build_tree
from embeddings.pipeline import build_embedding_indexes
from retrieval.pipeline import evaluate_index, load_queries_from_file

@task
def build_tree_task(md_path):
    return build_tree(md_path=md_path)

@task
def build_embeddings_task(tree_json):
    return build_embedding_indexes(tree_json=tree_json, which="both")

@task
def evaluate_task(title_coll, text_coll, queries):
    return evaluate_index(
        queries=queries,
        title_collection=title_coll,
        text_collection=text_coll,
    )

@flow(name="exrag-pipeline")
def exrag_flow(md_path, queries_path):
    tree_result = build_tree_task(md_path)
    embed_result = build_embeddings_task(tree_result.output_path)
    
    queries = load_queries_from_file(queries_path)
    metrics = evaluate_task(
        embed_result.title_stats.collection_name,
        embed_result.text_stats.collection_name,
        queries,
    )
    
    return metrics

# Run
if __name__ == "__main__":
    metrics = pageindex_flow("md/biology.md", "queries.txt")
    print(f"Success rate: {metrics.success_rate:.1%}")
```

See `PIPELINE.md` for advanced orchestration patterns (change detection, partitions, sensors).

## 📊 Data Models

All pipeline stages use Pydantic models for type safety:

```python
from common.models import (
    TreeBuildResult,      # Tree building result
    EmbeddingRunResult,   # Embedding result
    IndexStats,           # Per-index statistics
    AgentResult,          # Query result
    EvalMetrics,          # Evaluation metrics
    EmbeddingConfig,      # Embedding configuration
    RetrievalConfig,      # Retrieval configuration
)

# All models support:
result.model_dump_json()  # Serialize to JSON
result.model_dump()       # Serialize to dict
```

## 🧪 Testing

```bash
# Test tree building
python -m tree build md/biology.md --verbose

# Test embedding building
python -m embeddings build results/biology_structure.json --verbose

# Test retrieval
python -m retrieval query "test query" \
  --title biology_titles --text biology_texts --verbose

# Full pipeline test
python exrag_cli.py pipeline md/biology.md --verbose

# Run evaluation
echo "What are electric charges?" > test_queries.txt
echo "Explain Newton's laws" >> test_queries.txt
python -m retrieval eval test_queries.txt \
  --title biology_titles --text biology_texts
```

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"
- Create `.env` file: `cp .env.example .env`
- Set `OPENAI_API_KEY=your_key` in `.env`

### "Collection not found"
- List collections: `python -m embeddings list`
- Use exact collection names (case-sensitive)

### "No nodes found in JSON"
- Verify tree JSON: `python md_to_tree.py print results/biology.json`
- Check markdown has headers: `# ## ### ...`

### Rate limit errors
- Reduce batch size: `--batch-size 50`
- Adjust `PAGEINDEX_EMBEDDING_MAX_RETRIES` in `.env`

## 📚 Documentation

- **README.md** (this file): Overview and quick start
- **PIPELINE.md**: Detailed pipeline guide, CLI reference, orchestration examples
- **common/models.py**: Data model definitions
- **common/settings.py**: Configuration options

## 🤝 Contributing

This pipeline is structured for modularity and extensibility:

- **Add new embedding models**: Update `embeddings/embedder.py`
- **Add new agent tools**: Update `retrieval/tools.py`
- **Add new CLI commands**: Update module CLIs or `exrag_cli.py`
- **Add new data models**: Update `common/models.py`

## 📝 License

[Your License Here]

## 🔗 Related Tools

- **Prefect**: Workflow orchestration (recommended)
- **Dagster**: Alternative orchestrator with asset graphs
- **LangGraph**: Agent framework (used in retrieval)
- **ChromaDB**: Vector database (embedded)
- **Typer**: CLI framework (used throughout)
- **Pydantic**: Data validation (used throughout)

