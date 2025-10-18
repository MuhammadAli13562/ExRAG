# C-Embeddings: Vector Indexing for JSON Trees

Vectorization system for creating ChromaDB collections from flat JSON tree files.

## Features

- **Atomic node embeddings**: One embedding per node, preserving full granularity
- **Rich metadata extraction**: Chapters, sections, page numbers, anchor types, heading levels
- **Batch processing**: Efficient OpenAI API usage with progress tracking
- **Unique collections**: Each JSON file gets its own named collection
- **Persistent storage**: ChromaDB with HNSW indexing and cosine similarity

## Setup

Install dependencies:

```bash
pip install chromadb openai tqdm
```

Set OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Index a JSON file

```bash
python run_indexer.py /path/to/physics_structure.json
```

### Custom collection name

```bash
python run_indexer.py /path/to/physics.json --collection-name physics_book
```

### Reset existing collection

```bash
python run_indexer.py /path/to/physics.json --reset
```

### List all collections

```bash
python run_indexer.py --list
```

### Get collection info

```bash
python run_indexer.py --info physics_structure_abc123
```

## Configuration

Edit `config.py` to customize:

- **Embedding model**: Default is `text-embedding-3-small` (1536 dims)
- **Batch size**: Default is 100 nodes per API call
- **Metadata patterns**: Regex for chapters, sections, pages, anchors
- **ChromaDB location**: Default is `./chroma_db/`

## Metadata Fields

Each vector is stored with:

- `node_id`: Original node identifier
- `title`: Node heading/title
- `line_num`: Line number in source
- `summary`: Node summary (if available)
- `chapter`: Extracted chapter number
- `section`: Extracted section number
- `page_num`: Primary page reference
- `all_pages`: All page numbers mentioned
- `anchor_type`: Type (Experiment, Discussion, etc.)
- `heading_level`: Estimated depth (1-6)
- `token_count`: Approximate token count

## Output

ChromaDB collections stored in `chroma_db/` with:

- HNSW index (cosine similarity)
- Full metadata for filtering
- Original text for retrieval
- Embedding vectors for semantic search

## Next Steps

After indexing, you can:

- Build graph reconstruction (hierarchy + semantic neighbors)
- Implement agentic retrieval with LangGraph
- Add sparse/BM25 index for hybrid search
- Create context assembly logic with neighbor expansion

