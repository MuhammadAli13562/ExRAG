# C-Embeddings: Atomic Node Vectorization System

## What This Does

Creates ChromaDB vector collections from flat JSON tree structures, with one embedding per atomic node.

## Architecture

```
JSON Tree File
      ↓
[Load & Validate]
      ↓
[Extract Metadata] ← chapters, sections, pages, anchors, heading levels
      ↓
[Batch Embedding] ← OpenAI text-embedding-3-small (1536 dims)
      ↓
[ChromaDB Index] ← HNSW + cosine similarity
      ↓
Persistent Collection (queryable)
```

## Key Components

### 1. `config.py`
- Embedding model configuration
- ChromaDB settings
- Batch processing parameters
- Metadata extraction patterns

### 2. `metadata_extractor.py`
- Extracts structured metadata from raw nodes
- Detects chapters, sections, page numbers
- Identifies anchor types (Experiment, Discussion, etc.)
- Estimates heading levels and token counts
- Validates node quality

### 3. `embedder.py`
- OpenAI API wrapper
- Batch embedding generation
- Automatic retry logic
- Error handling

### 4. `indexer.py`
- Main orchestration class
- JSON loading with flexible schema support
- ChromaDB collection management
- Batch processing with progress tracking
- Statistics and reporting

### 5. `run_indexer.py` (CLI)
- Command-line interface
- Index files, list collections, get info
- Custom collection names
- Reset/overwrite support

### 6. `validate_structure.py`
- Pre-indexing validation
- Structure analysis and statistics
- No API key required

### 7. `test_index.py`
- Complete end-to-end test
- Uses physics_structure.json
- Validates setup and reports results

## Metadata Stored Per Vector

Each embedded node includes:

```python
{
    "node_id": "0850",
    "title": "Experiment 7a",
    "line_num": 10948,
    "summary": "...",
    "chapter": "7",
    "section": "",
    "anchor_type": "Experiment",
    "page_num": 269,
    "all_pages": "269",
    "heading_level": 2,
    "token_count": 145
}
```

## Design Decisions

### Why Atomic Nodes?

1. **Maximum granularity** - Preserve all original structure
2. **Flexibility** - Assembly/stitching happens at retrieval time
3. **Simplicity** - No complex segmentation upfront
4. **Graph-ready** - Graph reconstruction can happen independently

### Why ChromaDB?

1. **Simple** - Easy setup, no server required
2. **Persistent** - Built-in storage
3. **Fast** - HNSW indexing for sub-millisecond queries
4. **Rich filters** - Metadata filtering built-in
5. **Python-native** - Easy integration

### Why OpenAI Embeddings?

1. **Quality** - State-of-the-art semantic understanding
2. **Cost** - text-embedding-3-small is very affordable ($0.02/1M tokens)
3. **Speed** - Batch API is efficient
4. **Reliability** - Production-grade infrastructure

## Usage Pattern

```bash
# 1. Validate structure
python validate_structure.py your_file.json

# 2. Index
python run_indexer.py your_file.json --collection-name my_book

# 3. Verify
python run_indexer.py --list
python run_indexer.py --info my_book
```

## Statistics (physics_structure.json)

- **Total nodes**: 1,118
- **Valid nodes**: 1,111 (7 filtered)
- **Chapters**: 28
- **Experiments**: 70
- **Examples**: 85
- **Avg tokens/node**: 121
- **Indexing time**: ~30-60 seconds
- **Cost**: < $0.01

## Next Steps

This forms the foundation for:

1. **Graph reconstruction** - Build hierarchy and semantic edges
2. **Agentic retrieval** - LangGraph-based planning and execution
3. **Hybrid search** - Add BM25/sparse index
4. **Context assembly** - Neighbor expansion and stitching
5. **Multi-resolution** - Add block-level embeddings

## Files Overview

| File | Purpose | Lines |
|------|---------|-------|
| `config.py` | Configuration | ~50 |
| `metadata_extractor.py` | Metadata extraction | ~120 |
| `embedder.py` | OpenAI API wrapper | ~60 |
| `indexer.py` | Main indexing logic | ~200 |
| `run_indexer.py` | CLI entry point | ~110 |
| `validate_structure.py` | Structure validator | ~90 |
| `test_index.py` | End-to-end test | ~70 |

**Total**: ~700 lines of clean, documented Python

## Performance

- **Embedding speed**: ~100 nodes/second (batched)
- **Memory**: ~50MB for 1,100 nodes
- **Query latency**: <10ms for top-k retrieval
- **Storage**: ~5MB per 1,000 vectors

## Extensibility

Easy to extend:

- **Add new metadata extractors** - Edit `metadata_extractor.py`
- **Change embedding model** - Edit `config.py` and `embedder.py`
- **Add preprocessing** - Extend `is_valid_node()` and node loading
- **Custom filters** - Use ChromaDB's rich metadata queries
- **Alternative backends** - Swap ChromaDB for Qdrant/Weaviate/Pinecone

