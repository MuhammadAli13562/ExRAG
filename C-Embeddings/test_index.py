#!/usr/bin/env python3
"""
Test script for indexing the physics_structure.json file.

This script demonstrates the complete indexing pipeline.
Make sure OPENAI_API_KEY is set before running.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from indexer import VectorIndexer


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Path to physics structure JSON
    project_root = Path(__file__).parent.parent
    physics_json = project_root / "results" / "physics_structure.json"
    
    if not physics_json.exists():
        print(f"ERROR: File not found: {physics_json}")
        return 1
    
    print("Starting vectorization test...")
    print(f"File: {physics_json}")
    print()
    
    # Create indexer and run pipeline
    indexer = VectorIndexer()
    
    try:
        stats = indexer.index_json_file(
            json_path=physics_json,
            collection_name="physics_structure_test",
            reset=True,  # Reset if exists
        )
        
        # Print final statistics
        print("\n" + "="*60)
        print("SUCCESS! Vectorization Complete")
        print("="*60)
        print(f"Collection: {stats['collection_name']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Indexed: {stats['indexed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['indexed']/stats['total_nodes']*100:.1f}%")
        print("="*60)
        
        # Show collection info
        info = indexer.get_collection_info(stats['collection_name'])
        print(f"\nCollection now contains {info['count']} vectors")
        print(f"Metadata: {info['metadata']}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

