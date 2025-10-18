#!/usr/bin/env python3
"""
Example usage of the agentic retrieval system.

This demonstrates how to use the agent programmatically.
"""
from agent import query_agent
from tools import RetrievalTools


def example_single_query():
    """Example: Single query with specific collections."""
    print("\n" + "="*80)
    print("Example 1: Single Query")
    print("="*80)
    
    # List available collections first
    tools = RetrievalTools()
    collections = tools.list_collections()
    
    # Find title and text collections (assuming physics document)
    title_coll = None
    text_coll = None
    
    for coll in collections:
        name = coll['name']
        index_field = coll['metadata'].get('index_field')
        
        if 'physics' in name.lower():
            if index_field == 'title':
                title_coll = name
            elif index_field == 'text':
                text_coll = name
    
    if not title_coll or not text_coll:
        print("Error: Could not find physics collections.")
        print("Available collections:")
        for coll in collections:
            print(f"  - {coll['name']} (index_field: {coll['metadata'].get('index_field')})")
        return
    
    print(f"\nUsing collections:")
    print(f"  Title: {title_coll}")
    print(f"  Text: {text_coll}")
    
    # Run query
    result = query_agent(
        user_query="Explain the two kinds of electric charges and how they interact.",
        title_collection=title_coll,
        text_collection=text_coll,
        verbose=True
    )
    
    return result


def example_exploration_focused():
    """Example: Query that requires exploration."""
    print("\n" + "="*80)
    print("Example 2: Exploration-Focused Query")
    print("="*80)
    
    tools = RetrievalTools()
    collections = tools.list_collections()
    
    # Find collections
    title_coll = None
    text_coll = None
    
    for coll in collections:
        name = coll['name']
        index_field = coll['metadata'].get('index_field')
        
        if 'physics' in name.lower():
            if index_field == 'title':
                title_coll = name
            elif index_field == 'text':
                text_coll = name
    
    if not title_coll or not text_coll:
        print("Error: Could not find physics collections.")
        return
    
    # This query should trigger exploration to get full context
    result = query_agent(
        user_query="What is the coulomb as a unit? Give me detailed information including examples.",
        title_collection=title_coll,
        text_collection=text_coll,
        verbose=True
    )
    
    return result


def example_direct_tool_usage():
    """Example: Using tools directly without agent."""
    print("\n" + "="*80)
    print("Example 3: Direct Tool Usage (No Agent)")
    print("="*80)
    
    tools = RetrievalTools()
    
    # List collections
    print("\n1. Listing collections...")
    collections = tools.list_collections()
    
    for coll in collections:
        print(f"  - {coll['name']}")
        print(f"    Index Field: {coll['metadata'].get('index_field')}")
        print(f"    Count: {coll['count']}")
        print()
    
    # Find collections
    title_coll = None
    text_coll = None
    
    for coll in collections:
        name = coll['name']
        index_field = coll['metadata'].get('index_field')
        
        if 'physics' in name.lower():
            if index_field == 'title':
                title_coll = name
            elif index_field == 'text':
                text_coll = name
    
    if not title_coll or not text_coll:
        print("Error: Could not find physics collections.")
        return
    
    # Search by title
    print("\n2. Searching by title...")
    title_results = tools.search_by_title(
        query="electric charges",
        collection_name=title_coll,
        top_k=3
    )
    
    print(f"Found {len(title_results)} results:")
    for result in title_results:
        print(f"  - {result['node_id']}: {result['title'][:60]}...")
        print(f"    Similarity: {result['similarity_score']:.3f}")
    
    # Search by text
    print("\n3. Searching by text...")
    text_results = tools.search_by_text(
        query="electric force between charges",
        collection_name=text_coll,
        top_k=3
    )
    
    print(f"Found {len(text_results)} results:")
    for result in text_results:
        print(f"  - {result['node_id']}: {result['text'][:60]}...")
        print(f"    Similarity: {result['similarity_score']:.3f}")
    
    # Explore around a node
    if title_results:
        target_node_id = title_results[0]['node_id']
        print(f"\n4. Exploring around node {target_node_id}...")
        
        exploration = tools.explore_nodes(
            node_id=target_node_id,
            collection_name=title_coll,
            direction="both",
            count=2
        )
        
        print(f"Target Node: {exploration['target_node']['title']}")
        print(f"Nodes Above: {len(exploration['nodes_above'])}")
        print(f"Nodes Below: {len(exploration['nodes_below'])}")
        print(f"Position: {exploration['position_info']['target_index']} / {exploration['position_info']['total_nodes']}")


def example_comparison():
    """Example: Compare different search strategies."""
    print("\n" + "="*80)
    print("Example 4: Comparing Search Strategies")
    print("="*80)
    
    tools = RetrievalTools()
    collections = tools.list_collections()
    
    # Find collections
    title_coll = None
    text_coll = None
    
    for coll in collections:
        name = coll['name']
        index_field = coll['metadata'].get('index_field')
        
        if 'physics' in name.lower():
            if index_field == 'title':
                title_coll = name
            elif index_field == 'text':
                text_coll = name
    
    if not title_coll or not text_coll:
        print("Error: Could not find physics collections.")
        return
    
    query = "Coulomb's law"
    
    # Title search
    print(f"\nQuery: '{query}'")
    print("\n--- Title Search ---")
    title_results = tools.search_by_title(query, title_coll, top_k=3)
    for i, result in enumerate(title_results, 1):
        print(f"{i}. [{result['node_id']}] {result['title'][:70]}...")
        print(f"   Score: {result['similarity_score']:.3f}")
    
    # Text search
    print("\n--- Text Search ---")
    text_results = tools.search_by_text(query, text_coll, top_k=3)
    for i, result in enumerate(text_results, 1):
        print(f"{i}. [{result['node_id']}] {result['text'][:70]}...")
        print(f"   Score: {result['similarity_score']:.3f}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("AGENTIC RETRIEVAL SYSTEM - EXAMPLES")
    print("="*80)
    
    # Check if collections exist
    tools = RetrievalTools()
    collections = tools.list_collections()
    
    if not collections:
        print("\n⚠️  No collections found!")
        print("\nPlease run the indexer first to create embeddings:")
        print("  cd ../embeddings")
        print("  python run_indexer.py ../results/physics_structure.json --index-field title")
        print("  python run_indexer.py ../results/physics_structure.json --index-field text")
        sys.exit(1)
    
    # Menu
    print("\nAvailable examples:")
    print("  1. Single query with agent")
    print("  2. Exploration-focused query")
    print("  3. Direct tool usage (no agent)")
    print("  4. Compare search strategies")
    print("  5. Run all examples")
    
    choice = input("\nSelect example (1-5): ").strip()
    
    if choice == "1":
        example_single_query()
    elif choice == "2":
        example_exploration_focused()
    elif choice == "3":
        example_direct_tool_usage()
    elif choice == "4":
        example_comparison()
    elif choice == "5":
        example_single_query()
        example_exploration_focused()
        example_direct_tool_usage()
        example_comparison()
    else:
        print("Invalid choice")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")

