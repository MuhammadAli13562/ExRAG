# Agentic Retrieval System

An intelligent retrieval system that uses LangGraph to orchestrate semantic search and context exploration over embedded document structures.

## Overview

This system provides an agentic approach to document retrieval that goes beyond simple vector search. The agent can:

- **Search by Title**: Find relevant sections based on semantic similarity to titles/headings
- **Search by Text**: Find content based on semantic similarity to the actual text
- **Explore Context**: Navigate the document structure to retrieve surrounding nodes for better context
- **Iterative Refinement**: Autonomously explore and gather information until it has comprehensive context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                         │
│                   (GPT-4o Orchestration)                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Title Search  │    │ Text Search  │    │   Explore    │
│  (ChromaDB)  │    │  (ChromaDB)  │    │    Nodes     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌──────────────┐
                    │Vector + JSON │
                    │   Storage    │
                    └──────────────┘
```

## Components

### 1. Tools (`tools.py`)
- `RetrievalTools`: Core retrieval functionality
  - `search_by_title()`: Semantic search on titles
  - `search_by_text()`: Semantic search on content
  - `explore_nodes()`: Navigate document structure
  - `list_collections()`: Discover available collections

### 2. Agent (`agent.py`)
- LangGraph-based orchestration
- Autonomous tool selection and execution
- Iterative refinement strategy
- Context synthesis

### 3. Configuration (`config.py`)
- Agent model settings (GPT-4o)
- Retrieval parameters
- Exploration limits

### 4. CLI (`run_agent.py`)
- Interactive mode
- Single query mode
- Collection management

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `langgraph` - Agent orchestration
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `chromadb` - Vector database
- `openai` - Embeddings API

## Usage

### List Available Collections

```bash
cd retrieval
python run_agent.py --list-collections
```

### Single Query Mode

```bash
python run_agent.py \
  --query "Explain the concept of electric charges" \
  --title-collection "physics_structure_HASH_title" \
  --text-collection "physics_structure_HASH_text"
```

### Interactive Mode

```bash
python run_agent.py \
  --interactive \
  --title-collection "physics_structure_HASH_title" \
  --text-collection "physics_structure_HASH_text"
```

### Programmatic Usage

```python
from retrieval.agent import query_agent

result = query_agent(
    user_query="What is Coulomb's law?",
    title_collection="physics_structure_abc123_title",
    text_collection="physics_structure_abc123_text",
    verbose=True
)

print(result['answer'])
print(f"Iterations: {result['iteration_count']}")
```

## How It Works

### Agent Strategy

1. **Initial Search**
   - Performs both title and text searches in parallel
   - Identifies top candidates based on similarity scores

2. **Context Exploration**
   - Selects promising nodes (high similarity scores)
   - Uses `explore_nodes` to get surrounding context
   - Can explore up (previous nodes) or down (next nodes)

3. **Iterative Refinement**
   - Continues exploring until sufficient context is gathered
   - May explore multiple branches if needed
   - Maximum iterations: 15 (configurable)

4. **Answer Synthesis**
   - Synthesizes comprehensive answer from all gathered context
   - Provides direct response to user's query

### Example Agent Flow

```
User Query: "How do electric charges interact?"
    │
    ├─> Title Search: "electric charges interaction"
    │   └─> Results: [node_0004 (0.85), node_0006 (0.78), ...]
    │
    ├─> Text Search: "electric charges interaction"
    │   └─> Results: [node_0006 (0.89), node_0010 (0.74), ...]
    │
    ├─> Explore node_0006 (highest score, both directions)
    │   └─> Context: nodes [0003-0009]
    │
    ├─> Explore node_0010 (additional context)
    │   └─> Context: nodes [0007-0013]
    │
    └─> Synthesize Final Answer
```

## Configuration

Edit `config.py` to customize:

```python
# Agent model
AGENT_MODEL = "gpt-4o"  # LLM for orchestration
TEMPERATURE = 0  # Deterministic

# Retrieval
DEFAULT_TOP_K = 5  # Results per search
SIMILARITY_THRESHOLD = 0.5  # Minimum score

# Exploration
MAX_EXPLORATION_DEPTH = 10  # Max nodes per direction
DEFAULT_EXPLORATION_COUNT = 3  # Default context size

# Limits
MAX_ITERATIONS = 15  # Max agent iterations
RECURSION_LIMIT = 25  # LangGraph limit
```

## Tool Descriptions

### search_by_title
```python
search_by_title(
    query: str,              # Search query
    collection_name: str,    # Title collection name
    top_k: int = 5          # Number of results
) -> str                    # JSON results
```

### search_by_text
```python
search_by_text(
    query: str,              # Search query
    collection_name: str,    # Text collection name
    top_k: int = 5          # Number of results
) -> str                    # JSON results
```

### explore_nodes
```python
explore_nodes(
    node_id: str,            # Target node ID (e.g., "0042")
    collection_name: str,    # Collection name
    direction: str = "both", # "up", "down", or "both"
    count: int = 3          # Nodes per direction
) -> str                    # JSON with context
```

## Advanced Features

### Multi-Document Support
The agent can work with multiple documents by using different collection pairs:

```python
# Physics document
result_physics = query_agent(
    user_query=query,
    title_collection="physics_structure_abc123_title",
    text_collection="physics_structure_abc123_text"
)

# Biology document
result_biology = query_agent(
    user_query=query,
    title_collection="biology_structure_xyz789_title",
    text_collection="biology_structure_xyz789_text"
)
```

### Custom Exploration Strategies
The agent autonomously decides when and how to explore, but you can influence its behavior through the query:

```python
# Request specific depth
query = "Explain electric fields in detail with surrounding context"

# Request specific sections
query = "Find the section on Coulomb's law and show what comes before it"
```

## Troubleshooting

### No results found
- Check collection names with `--list-collections`
- Verify embeddings were created for the JSON file
- Try broader query terms

### Agent not exploring enough
- Increase `MAX_ITERATIONS` in config.py
- Use more specific queries that indicate need for context

### Out of memory
- Reduce `DEFAULT_TOP_K` to return fewer results
- Decrease `MAX_EXPLORATION_DEPTH`

## Performance Tips

1. **Be Specific**: More specific queries lead to better retrieval
2. **Use Both Searches**: Agent uses both title and text - leverage this
3. **Context Matters**: The agent excels when context is important
4. **Iteration Monitoring**: Watch iteration counts to optimize

## Future Enhancements

- [ ] Streaming responses for real-time output
- [ ] Citation tracking for source attribution
- [ ] Multi-document cross-referencing
- [ ] User feedback loop for refinement
- [ ] Caching for repeated queries
- [ ] Visualization of agent decision tree

## License

Part of the PageIndex project.

