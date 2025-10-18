#!/usr/bin/env python3
"""
Tests for the agentic retrieval system.
Run this to verify the system is working correctly.
"""
import sys
from pathlib import Path

from tools import RetrievalTools
from agent import query_agent


def test_tools_initialization():
    """Test that tools can be initialized."""
    print("\n" + "="*60)
    print("Test 1: Tools Initialization")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        print("✅ RetrievalTools initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize tools: {e}")
        return False


def test_list_collections():
    """Test listing collections."""
    print("\n" + "="*60)
    print("Test 2: List Collections")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        collections = tools.list_collections()
        
        print(f"Found {len(collections)} collection(s)")
        for coll in collections:
            print(f"  - {coll['name']}")
            print(f"    Count: {coll['count']}")
            print(f"    Index Field: {coll['metadata'].get('index_field', 'N/A')}")
        
        if len(collections) == 0:
            print("⚠️  Warning: No collections found")
            print("   Please run indexer first to create embeddings")
            return False
        
        print("✅ Collections listed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")
        return False


def test_title_search():
    """Test title-based search."""
    print("\n" + "="*60)
    print("Test 3: Title Search")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        collections = tools.list_collections()
        
        # Find a title collection
        title_coll = None
        for coll in collections:
            if coll['metadata'].get('index_field') == 'title':
                title_coll = coll['name']
                break
        
        if not title_coll:
            print("⚠️  No title-indexed collection found")
            return False
        
        print(f"Using collection: {title_coll}")
        
        # Perform search
        results = tools.search_by_title(
            query="electric charges",
            collection_name=title_coll,
            top_k=3
        )
        
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - {result['node_id']}: {result.get('title', 'N/A')[:50]}...")
            print(f"    Score: {result['similarity_score']:.3f}")
        
        if len(results) > 0:
            print("✅ Title search successful")
            return True
        else:
            print("⚠️  Warning: No results found")
            return False
        
    except Exception as e:
        print(f"❌ Failed title search: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_search():
    """Test text-based search."""
    print("\n" + "="*60)
    print("Test 4: Text Search")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        collections = tools.list_collections()
        
        # Find a text collection
        text_coll = None
        for coll in collections:
            if coll['metadata'].get('index_field') == 'text':
                text_coll = coll['name']
                break
        
        if not text_coll:
            print("⚠️  No text-indexed collection found")
            return False
        
        print(f"Using collection: {text_coll}")
        
        # Perform search
        results = tools.search_by_text(
            query="electric force between charges",
            collection_name=text_coll,
            top_k=3
        )
        
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - {result['node_id']}: {result.get('text', 'N/A')[:50]}...")
            print(f"    Score: {result['similarity_score']:.3f}")
        
        if len(results) > 0:
            print("✅ Text search successful")
            return True
        else:
            print("⚠️  Warning: No results found")
            return False
        
    except Exception as e:
        print(f"❌ Failed text search: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_exploration():
    """Test node exploration."""
    print("\n" + "="*60)
    print("Test 5: Node Exploration")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        collections = tools.list_collections()
        
        # Find any collection
        coll_name = None
        for coll in collections:
            coll_name = coll['name']
            break
        
        if not coll_name:
            print("⚠️  No collections found")
            return False
        
        print(f"Using collection: {coll_name}")
        
        # First search to get a node_id
        if collections[0]['metadata'].get('index_field') == 'title':
            results = tools.search_by_title("electric", coll_name, top_k=1)
        else:
            results = tools.search_by_text("electric", coll_name, top_k=1)
        
        if not results:
            print("⚠️  No results to explore")
            return False
        
        node_id = results[0]['node_id']
        print(f"Exploring around node: {node_id}")
        
        # Explore
        exploration = tools.explore_nodes(
            node_id=node_id,
            collection_name=coll_name,
            direction="both",
            count=2
        )
        
        print(f"Target node: {exploration['target_node']['title']}")
        print(f"Nodes above: {len(exploration['nodes_above'])}")
        print(f"Nodes below: {len(exploration['nodes_below'])}")
        print(f"Position: {exploration['position_info']['target_index']} / {exploration['position_info']['total_nodes']}")
        
        print("✅ Node exploration successful")
        return True
        
    except Exception as e:
        print(f"❌ Failed node exploration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_query():
    """Test full agent query."""
    print("\n" + "="*60)
    print("Test 6: Agent Query (Full System)")
    print("="*60)
    
    try:
        tools = RetrievalTools()
        collections = tools.list_collections()
        
        # Find title and text collections
        title_coll = None
        text_coll = None
        
        for coll in collections:
            index_field = coll['metadata'].get('index_field')
            name = coll['name']
            
            # Try to find matching pairs
            if index_field == 'title' and not title_coll:
                title_coll = name
            elif index_field == 'text' and not text_coll:
                text_coll = name
        
        if not title_coll or not text_coll:
            print("⚠️  Need both title and text collections")
            print(f"   Title collection: {title_coll or 'NOT FOUND'}")
            print(f"   Text collection: {text_coll or 'NOT FOUND'}")
            return False
        
        print(f"Using collections:")
        print(f"  Title: {title_coll}")
        print(f"  Text: {text_coll}")
        print()
        
        # Run a simple query
        result = query_agent(
            user_query="What are the two kinds of electric charges?",
            title_collection=title_coll,
            text_collection=text_coll,
            verbose=False  # Don't print detailed output in test
        )
        
        print(f"\nAgent completed in {result['iteration_count']} iterations")
        print(f"Answer length: {len(result['answer'])} characters")
        print(f"\nAnswer preview:")
        print("-" * 60)
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        print("-" * 60)
        
        if result['answer'] and len(result['answer']) > 50:
            print("✅ Agent query successful")
            return True
        else:
            print("⚠️  Warning: Answer seems incomplete")
            return False
        
    except Exception as e:
        print(f"❌ Failed agent query: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("AGENTIC RETRIEVAL SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Tools Initialization", test_tools_initialization),
        ("List Collections", test_list_collections),
        ("Title Search", test_title_search),
        ("Text Search", test_text_search),
        ("Node Exploration", test_node_exploration),
        ("Agent Query", test_agent_query),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

