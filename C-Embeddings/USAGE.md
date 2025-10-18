# C-Embeddings Usage Guide

## Quick Start

### 1. Validate your JSON file first

```bash
python validate_structure.py ../results/physics_structure.json
```

This will show you:
- Total nodes and valid/invalid counts
- Chapters and sections found
- Anchor types distribution
- Token statistics

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Index your JSON file

**Simple usage (auto-generated collection name):**
```bash
python run_indexer.py ../results/physics_structure.json
```

**Custom collection name:**
```bash
python run_indexer.py ../results/physics_structure.json --collection-name my_physics_book
```

**Reset existing collection:**
```bash
python run_indexer.py ../results/physics_structure.json --reset
```

### 4. Verify the collection

**List all collections:**
```bash
python run_indexer.py --list
```

**Get collection info:**
```bash
python run_indexer.py --info physics_structure_abc12345
```

## Using the Test Script

Run the complete test with physics_structure.json:

```bash
python test_index.py
```

This will:
- Validate API key is set
- Load the physics JSON
- Create a test collection
- Index all nodes with progress tracking
- Show final statistics

## Expected Output

For physics_structure.json, you should see:

```
Loaded 1118 nodes from physics_structure.json
Filtered out 7 empty/trivial nodes

Indexing 1111 nodes into collection 'physics_structure_test'
Batch size: 100

Embedding batches: 100% ███████████ 12/12

============================================================
Indexing Complete
============================================================
Collection: physics_structure_test
Total nodes: 1111
Indexed: 1111
Failed: 0
Batches: 12
Success rate: 100.0%
============================================================

Collection now contains 1111 vectors
```

## Cost Estimation

For `text-embedding-3-small`:
- Price: $0.02 per 1M tokens
- Physics book: ~1,111 nodes × ~121 tokens avg = ~134,000 tokens
- Cost: ~$0.003 (less than a penny)

## Troubleshooting

**"OPENAI_API_KEY not set"**
- Make sure you exported the key in your current shell
- Check: `echo $OPENAI_API_KEY`

**"Unexpected JSON structure"**
- Run `validate_structure.py` to check your JSON format
- Supported keys: `structure`, `nodes`, `data`, `items`
- Or use a plain array of nodes

**Rate limit errors**
- The script has automatic retry logic
- If persistent, reduce `BATCH_SIZE` in `config.py`

**Empty collection after indexing**
- Check if nodes are being filtered out (run validation script)
- Look for error messages during embedding batches

## Next Steps

After indexing, you can:

1. **Query the collection** (coming next in D-Retrieval):
   ```python
   from indexer import VectorIndexer
   indexer = VectorIndexer()
   collection = indexer.client.get_collection("physics_structure_test")
   
   # Query example
   results = collection.query(
       query_texts=["How does electromagnetic induction work?"],
       n_results=10
   )
   ```

2. **Build graph reconstruction** for context assembly

3. **Implement agentic retrieval** with LangGraph

4. **Add hybrid search** (dense + sparse)

