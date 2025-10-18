#!/usr/bin/env python3
"""
CLI entry point for vectorizing JSON tree files.

Usage:
    python run_indexer.py <json_path> [--collection-name NAME] [--reset]
    python run_indexer.py --list
    python run_indexer.py --info COLLECTION_NAME
"""
import argparse
import sys
from pathlib import Path

from indexer import VectorIndexer


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize JSON tree files into ChromaDB collections"
    )
    
    # Main commands
    parser.add_argument(
        "json_path",
        nargs='?',
        type=Path,
        help="Path to JSON tree file to index"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Custom collection name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before indexing"
    )
    parser.add_argument(
        "--index-field",
        type=str,
        default="text",
        choices=["text", "title", "summary", "prefix_summary"],
        help="Which field to index (default: text). Options: text, title, summary, prefix_summary"
    )
    
    # Info commands
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing collections"
    )
    parser.add_argument(
        "--info",
        type=str,
        metavar="COLLECTION_NAME",
        help="Show information about a collection"
    )
    
    args = parser.parse_args()
    
    indexer = VectorIndexer()
    
    # Handle info commands
    if args.list:
        collections = indexer.list_collections()
        print(f"\nFound {len(collections)} collection(s):")
        for name in collections:
            info = indexer.get_collection_info(name)
            print(f"  - {name} ({info.get('count', 0)} vectors)")
        return 0
    
    if args.info:
        info = indexer.get_collection_info(args.info)
        if "error" in info:
            print(f"Error: {info['error']}")
            return 1
        print(f"\nCollection: {info['name']}")
        print(f"Vector count: {info['count']}")
        print(f"Metadata: {info['metadata']}")
        return 0
    
    # Main indexing command
    if not args.json_path:
        parser.print_help()
        return 1
    
    try:
        stats = indexer.index_json_file(
            json_path=args.json_path,
            collection_name=args.collection_name,
            reset=args.reset,
            index_field=args.index_field,
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("Indexing Complete")
        print(f"{'='*60}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Index Field: {stats.get('index_field', 'text')}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Indexed: {stats['indexed']}")
        print(f"Skipped: {stats.get('skipped', 0)} (empty/missing field)")
        print(f"Failed: {stats['failed']}")
        print(f"Batches: {stats['batches']}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

