#!/usr/bin/env python3
"""
Example: Query an indexed collection.

This demonstrates basic querying capabilities before building the full agentic system.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from indexer import VectorIndexer


def example_queries(collection_name: str):
    """Run example queries against a collection."""
    
    indexer = VectorIndexer()
    
    # Get collection
    try:
        collection = indexer.client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found")
        print(f"\nAvailable collections: {indexer.list_collections()}")
        return
    
    print(f"Querying collection: {collection_name}")
    print(f"Total vectors: {collection.count()}\n")
    print("="*70)
    
    # Example queries
    queries = [
        "How does electromagnetic induction work?",
        "What is an armature in a motor?",
        "Experiment about moving magnet and coil",
        "Electric charges and electrostatics",
        "Page 239",
    ]
    
    for i, query_text in enumerate(queries, 1):
        print(f"\n{i}. Query: \"{query_text}\"")
        print("-" * 70)
        
        # Search
        results = collection.query(
            query_texts=[query_text],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        # Display results
        for j, (doc, metadata, distance) in enumerate(
            zip(results['documents'][0], results['metadatas'][0], results['distances'][0]),
            1
        ):
            print(f"\n  Result {j} (similarity: {1-distance:.3f}):")
            print(f"  Node ID: {metadata['node_id']}")
            print(f"  Title: {metadata['title']}")
            print(f"  Chapter: {metadata.get('chapter', 'N/A')}")
            print(f"  Page: {metadata.get('page_num', 'N/A')}")
            print(f"  Anchor: {metadata.get('anchor_type', 'N/A')}")
            print(f"  Line: {metadata['line_num']}")
            
            # Show snippet
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"  Text: {snippet}")
        
        print()
    
    print("="*70)
    print("\nQuery Examples Complete!")
    print("\nNext steps:")
    print("- Build graph reconstruction for context assembly")
    print("- Add neighbor expansion (±N nodes)")
    print("- Implement hybrid search (dense + BM25)")
    print("- Create agentic retrieval with LangGraph")


def example_metadata_filtering(collection_name: str):
    """Example: Filter by metadata."""
    
    indexer = VectorIndexer()
    collection = indexer.client.get_collection(collection_name)
    
    print("\n" + "="*70)
    print("Metadata Filtering Examples")
    print("="*70)
    
    # Example 1: Find all Experiments
    print("\n1. All Experiment nodes:")
    results = collection.query(
        query_texts=["experiment procedure"],
        n_results=5,
        where={"anchor_type": "Experiment"}
    )
    
    for metadata in results['metadatas'][0]:
        print(f"  - {metadata['title']} (Node {metadata['node_id']}, Line {metadata['line_num']})")
    
    # Example 2: Find nodes in Chapter 7
    print("\n2. Nodes in Chapter 7:")
    results = collection.query(
        query_texts=["electromagnetic"],
        n_results=5,
        where={"chapter": "7"}
    )
    
    for metadata in results['metadatas'][0]:
        print(f"  - {metadata['title']} (Node {metadata['node_id']})")
    
    # Example 3: Find page references
    print("\n3. Content from page 239:")
    results = collection.query(
        query_texts=["motor"],
        n_results=3,
        where={"page_num": 239}
    )
    
    for metadata, doc in zip(results['metadatas'][0], results['documents'][0]):
        print(f"  - {metadata['title']}")
        print(f"    {doc[:150]}...")


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_query.py <collection_name>")
        print("\nOr run with default test collection:")
        collection_name = "physics_structure_test"
        print(f"Using: {collection_name}\n")
    else:
        collection_name = sys.argv[1]
    
    try:
        # Run basic queries
        example_queries(collection_name)
        
        # Run metadata filtering examples
        example_metadata_filtering(collection_name)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

