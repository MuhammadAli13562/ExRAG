#!/usr/bin/env python3
"""
CLI entry point for the agentic retrieval system.

Usage:
    python run_agent.py --query "Your question here" --title-collection COLLECTION --text-collection COLLECTION
    python run_agent.py --interactive --title-collection COLLECTION --text-collection COLLECTION
    python run_agent.py --list-collections
"""
import argparse
import sys
from pathlib import Path

from agent import query_agent
from tools import RetrievalTools


def interactive_mode(title_collection: str, text_collection: str):
    """Run the agent in interactive mode."""
    print("\n" + "="*80)
    print("AGENTIC RETRIEVAL SYSTEM - Interactive Mode")
    print("="*80)
    print(f"Title Collection: {title_collection}")
    print(f"Text Collection: {text_collection}")
    print("\nType 'exit' or 'quit' to end the session")
    print("="*80 + "\n")
    
    while True:
        try:
            user_input = input("\n🔍 Your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            # Query the agent
            result = query_agent(
                user_query=user_input,
                title_collection=title_collection,
                text_collection=text_collection,
                verbose=True
            )
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def single_query_mode(query: str, title_collection: str, text_collection: str):
    """Run the agent for a single query."""
    result = query_agent(
        user_query=query,
        title_collection=title_collection,
        text_collection=text_collection,
        verbose=True
    )
    return result


def list_collections_mode():
    """List all available collections."""
    tools = RetrievalTools()
    collections = tools.list_collections()
    
    print("\n" + "="*80)
    print("AVAILABLE COLLECTIONS")
    print("="*80 + "\n")
    
    if not collections:
        print("No collections found.\n")
        return
    
    # Group by index field
    title_collections = []
    text_collections = []
    other_collections = []
    
    for coll in collections:
        index_field = coll['metadata'].get('index_field', 'unknown')
        if index_field == 'title':
            title_collections.append(coll)
        elif index_field == 'text':
            text_collections.append(coll)
        else:
            other_collections.append(coll)
    
    if title_collections:
        print("📚 Title-Indexed Collections:")
        for coll in title_collections:
            print(f"  - {coll['name']}")
            print(f"    Vectors: {coll['count']}")
            print(f"    Source: {coll['metadata'].get('json_source', 'N/A')}")
            print()
    
    if text_collections:
        print("📄 Text-Indexed Collections:")
        for coll in text_collections:
            print(f"  - {coll['name']}")
            print(f"    Vectors: {coll['count']}")
            print(f"    Source: {coll['metadata'].get('json_source', 'N/A')}")
            print()
    
    if other_collections:
        print("🔧 Other Collections:")
        for coll in other_collections:
            print(f"  - {coll['name']}")
            print(f"    Vectors: {coll['count']}")
            print(f"    Index Field: {coll['metadata'].get('index_field', 'unknown')}")
            print()
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Retrieval System - Intelligent document search and exploration"
    )
    
    # Main arguments
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--title-collection",
        type=str,
        help="Name of the title-indexed collection"
    )
    parser.add_argument(
        "--text-collection",
        type=str,
        help="Name of the text-indexed collection"
    )
    
    # Info commands
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available collections"
    )
    
    args = parser.parse_args()
    
    # Handle list collections
    if args.list_collections:
        list_collections_mode()
        return 0
    
    # Validate required arguments for query mode
    if not args.title_collection or not args.text_collection:
        parser.error("Both --title-collection and --text-collection are required (unless using --list-collections)")
    
    # Run in appropriate mode
    try:
        if args.interactive:
            interactive_mode(args.title_collection, args.text_collection)
        elif args.query:
            single_query_mode(args.query, args.title_collection, args.text_collection)
        else:
            parser.error("Either --query or --interactive must be specified")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

