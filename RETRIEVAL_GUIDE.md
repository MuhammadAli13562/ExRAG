# Agentic Retrieval System - Complete Guide

This guide walks you through the complete workflow from document processing to intelligent retrieval.

## Overview

The PageIndex project now includes a sophisticated agentic retrieval system that goes beyond simple vector search. It combines:

1. **Dual Embeddings**: Both title-based and text-based semantic search
2. **Contextual Exploration**: Navigate document structure to gather surrounding context
3. **Autonomous Agent**: LangGraph-powered agent that decides when and how to explore
4. **Intelligent Synthesis**: Combines multiple sources to provide comprehensive answers

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Document Processing                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PageIndex: Markdown → JSON Structure (pageindex/)              │
│  - Hierarchical decomposition                                   │
│  - Node extraction with metadata                                │
│  Output: results/physics_structure.json                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Embeddings: JSON → Vector Collections (embeddings/)            │
│  - Create title-based embeddings (ChromaDB)                     │
│  - Create text-based embeddings (ChromaDB)                      │
│  Output: Two collections per document                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Retrieval: Agentic Search & Exploration (retrieval/)           │
│  - Title search tool                                            │
│  - Text search tool                                             │
│  - Node exploration tool                                        │
│  - LangGraph orchestration                                      │
│  Output: Comprehensive answers with context                     │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Workflow

### Step 1: Process Document with PageIndex

Convert your markdown document into a structured JSON format:

```bash
# From project root
python run_pageindex.py --md_file md/physics.md --output results/physics_structure.json
```

This creates a JSON file with flattened nodes, each containing:
- `node_id`: Unique identifier (e.g., "0042")
- `title`: Section heading
- `text`: Full content including title
- `line_num`: Source line number
- `summary`: Content summary

### Step 2: Create Embeddings

Create two types of embeddings for semantic search:

```bash
# Navigate to embeddings directory
cd embeddings

# Create title-based embeddings
python run_indexer.py ../results/physics_structure.json --index-field title --reset

# Create text-based embeddings
python run_indexer.py ../results/physics_structure.json --index-field text --reset
```

This creates two ChromaDB collections:
- `physics_structure_XXXXXXXX_title` - Indexed on titles
- `physics_structure_XXXXXXXX_text` - Indexed on content

**Note**: The `XXXXXXXX` is a hash generated from the file path. Use `--list` to see actual names.

### Step 3: Verify Collections

Check that collections were created successfully:

```bash
python run_indexer.py --list
```

You should see both title and text collections with their vector counts.

### Step 4: Run Agentic Retrieval

Now you can query the system using the intelligent agent:

```bash
# Navigate to retrieval directory
cd ../retrieval

# List available collections (if you need collection names)
python run_agent.py --list-collections

# Interactive mode (recommended for exploration)
python run_agent.py --interactive \
  --title-collection "physics_structure_XXXXXXXX_title" \
  --text-collection "physics_structure_XXXXXXXX_text"

# Single query mode
python run_agent.py \
  --query "Explain Coulomb's law and its applications" \
  --title-collection "physics_structure_XXXXXXXX_title" \
  --text-collection "physics_structure_XXXXXXXX_text"
```

### Step 5: Test the System

Run the test suite to verify everything works:

```bash
cd retrieval
python test_retrieval.py
```

This will run comprehensive tests on all components.

## How the Agent Works

### Retrieval Strategy

The agent follows this autonomous strategy:

1. **Initial Semantic Search**
   - Searches both title and text collections
   - Identifies top candidates based on similarity scores
   
2. **Smart Exploration**
   - Selects promising nodes (high similarity)
   - Uses `explore_nodes` to get surrounding context
   - Can explore "up" (previous sections), "down" (next sections), or "both"
   
3. **Iterative Refinement**
   - Continues exploring until sufficient context is gathered
   - May explore multiple nodes if needed
   - Respects iteration limits (default: 15)
   
4. **Answer Synthesis**
   - Combines information from all retrieved chunks
   - Provides comprehensive, well-structured response

### Example Agent Workflow

```
User Query: "What is the coulomb and how large is it?"

Iteration 1:
  └─ search_by_title("coulomb unit", ..., top_k=5)
      → Results: [node_0009 (0.88), node_0010 (0.72), ...]

Iteration 2:
  └─ search_by_text("coulomb measurement examples", ..., top_k=5)
      → Results: [node_0009 (0.91), node_0010 (0.78), ...]

Iteration 3:
  └─ explore_nodes(node_id="0009", direction="both", count=3)
      → Context: nodes [0006, 0007, 0008, 0009, 0010, 0011, 0012]

Iteration 4:
  └─ [Agent synthesizes final answer from all gathered context]

Final Answer: "The coulomb (C) is the SI unit of electric charge..."
```

## Advanced Usage

### Programmatic API

```python
from retrieval.agent import query_agent

result = query_agent(
    user_query="Your question here",
    title_collection="physics_structure_abc123_title",
    text_collection="physics_structure_abc123_text",
    verbose=True
)

print(result['answer'])
print(f"Completed in {result['iteration_count']} iterations")
```

### Direct Tool Usage

For more control, use tools directly:

```python
from retrieval.tools import RetrievalTools

tools = RetrievalTools()

# Title search
title_results = tools.search_by_title(
    query="electric charges",
    collection_name="physics_structure_abc123_title",
    top_k=5
)

# Text search
text_results = tools.search_by_text(
    query="coulomb's law formula",
    collection_name="physics_structure_abc123_text",
    top_k=5
)

# Explore context
context = tools.explore_nodes(
    node_id="0042",
    collection_name="physics_structure_abc123_title",
    direction="both",
    count=3
)
```

### Configuration

Customize behavior in `retrieval/config.py`:

```python
# Agent model
AGENT_MODEL = "gpt-4o"  # Can use gpt-4, gpt-3.5-turbo, etc.
TEMPERATURE = 0  # 0 for deterministic, >0 for creative

# Retrieval
DEFAULT_TOP_K = 5  # Results per search
SIMILARITY_THRESHOLD = 0.5  # Minimum score

# Exploration
MAX_EXPLORATION_DEPTH = 10  # Max nodes per direction
DEFAULT_EXPLORATION_COUNT = 3  # Default context size

# Limits
MAX_ITERATIONS = 15  # Max agent iterations
```

## Multi-Document Support

Process multiple documents and query them separately:

```bash
# Process biology document
python run_pageindex.py --md_file md/biology.md --output results/biology_structure.json

# Create embeddings
cd embeddings
python run_indexer.py ../results/biology_structure.json --index-field title
python run_indexer.py ../results/biology_structure.json --index-field text

# Query biology document
cd ../retrieval
python run_agent.py --interactive \
  --title-collection "biology_structure_XXXXXXXX_title" \
  --text-collection "biology_structure_XXXXXXXX_text"
```

## Troubleshooting

### No collections found

**Problem**: `run_agent.py --list-collections` shows no collections

**Solution**: 
```bash
cd embeddings
python run_indexer.py --list  # Check if collections exist
# If not, create them:
python run_indexer.py ../results/physics_structure.json --index-field title
python run_indexer.py ../results/physics_structure.json --index-field text
```

### OpenAI API errors

**Problem**: API key errors or rate limits

**Solution**:
- Check `.env` file has valid `OPENAI_API_KEY`
- Verify API key has sufficient credits
- Reduce `DEFAULT_TOP_K` in config to make fewer API calls

### Agent not exploring enough

**Problem**: Answers lack context

**Solution**:
- Increase `MAX_ITERATIONS` in `retrieval/config.py`
- Use more specific queries that indicate need for context
- Manually explore using `explore_nodes` tool

### Low similarity scores

**Problem**: Retrieved chunks have low relevance

**Solution**:
- Try both title and text search separately
- Rephrase query to match document language
- Verify embeddings were created correctly
- Check if query matches document domain

## Performance Tips

1. **Start Broad**: Use general queries for initial exploration
2. **Be Specific**: More specific queries → better retrieval
3. **Use Both Searches**: Agent leverages both title and text
4. **Monitor Iterations**: High iteration count may indicate unclear query
5. **Cache Results**: For repeated queries, consider caching

## Example Queries

### Good Queries (Specific, Clear Intent)

- "What is Coulomb's law and how is it used?"
- "Explain the two types of electric charges"
- "How do you calculate electric field strength?"
- "What are the safety precautions for static electricity?"

### Less Effective Queries

- "Electricity" (too broad)
- "Chapter 1" (agent won't know what that means)
- "The formula" (which formula?)

## Next Steps

1. **Run Examples**: `cd retrieval && python example.py`
2. **Test System**: `python test_retrieval.py`
3. **Interactive Mode**: Start exploring with `run_agent.py --interactive`
4. **Customize**: Modify `config.py` for your use case
5. **Scale**: Process additional documents

## Project Structure

```
PageIndex/
├── md/                          # Source markdown files
├── results/                     # Processed JSON structures
├── pageindex/                   # Document processing
├── embeddings/                  # Vector indexing
│   ├── indexer.py              # Embedding creation
│   ├── run_indexer.py          # CLI for indexing
│   └── chroma_db/              # Vector storage
├── retrieval/                   # Agentic retrieval
│   ├── agent.py                # LangGraph agent
│   ├── tools.py                # Retrieval tools
│   ├── run_agent.py            # CLI for queries
│   ├── test_retrieval.py       # Test suite
│   └── example.py              # Usage examples
└── requirements.txt             # Python dependencies
```

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings and agent)
- ~100MB disk space per document (for embeddings)

## Support

For issues or questions:
1. Run test suite: `python retrieval/test_retrieval.py`
2. Check collection status: `python embeddings/run_indexer.py --list`
3. Verify configuration: `cat retrieval/config.py`

---

Built with LangGraph, ChromaDB, and OpenAI.

