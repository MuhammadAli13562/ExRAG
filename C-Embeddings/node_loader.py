"""
Load and lookup full nodes from JSON source files by node_id.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


def flatten_structure(structure: Any, parent_path: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Recursively flatten tree structure into a dict mapping node_id -> full_node.
    
    Args:
        structure: Tree structure (dict with 'nodes' or list of dicts)
        parent_path: Internal use for tracking hierarchy
        
    Returns:
        Dictionary mapping node_id to complete node data
    """
    node_map = {}
    
    if isinstance(structure, dict):
        # Single node
        if 'node_id' in structure:
            node_id = structure['node_id']
            # Store full node (excluding nested 'nodes' to avoid duplication)
            node_copy = {k: v for k, v in structure.items() if k != 'nodes'}
            node_map[node_id] = node_copy
        
        # Process children
        if 'nodes' in structure and structure['nodes']:
            child_nodes = flatten_structure(structure['nodes'], parent_path)
            node_map.update(child_nodes)
    
    elif isinstance(structure, list):
        # List of nodes
        for item in structure:
            child_nodes = flatten_structure(item, parent_path)
            node_map.update(child_nodes)
    
    return node_map


def load_json_structure(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file and return the raw structure.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def build_node_lookup(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a complete node lookup dictionary from JSON file.
    
    Args:
        json_path: Path to JSON structure file
        
    Returns:
        Dictionary mapping node_id to full node data
    """
    data = load_json_structure(json_path)
    
    # Handle different JSON structures
    if isinstance(data, list):
        structure = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['structure', 'nodes', 'data', 'items']:
            if key in data:
                structure = data[key]
                break
        else:
            # Treat whole dict as single structure
            structure = data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")
    
    # Flatten to node_id -> node mapping
    node_map = flatten_structure(structure)
    
    return node_map


def get_node_by_id(node_map: Dict[str, Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a node by its node_id.
    
    Args:
        node_map: Node lookup dictionary
        node_id: The node_id to find
        
    Returns:
        Full node data or None if not found
    """
    return node_map.get(node_id)


def get_node_field(node: Dict[str, Any], field: str, default: str = "") -> str:
    """
    Safely get a field from a node.
    
    Args:
        node: Node dictionary
        field: Field name to retrieve
        default: Default value if field not found
        
    Returns:
        Field value or default
    """
    return node.get(field, default)


# Cache for loaded node maps to avoid reloading same file
_node_map_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}


def get_cached_node_map(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Get node map with caching to avoid reloading same file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Cached or newly loaded node map
    """
    json_path = str(Path(json_path).absolute())
    
    if json_path not in _node_map_cache:
        _node_map_cache[json_path] = build_node_lookup(json_path)
    
    return _node_map_cache[json_path]


def clear_cache():
    """Clear the node map cache."""
    global _node_map_cache
    _node_map_cache = {}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python node_loader.py <json_file> [node_id]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    node_map = build_node_lookup(json_file)
    
    print(f"Loaded {len(node_map)} nodes from {json_file}")
    
    if len(sys.argv) > 2:
        node_id = sys.argv[2]
        node = get_node_by_id(node_map, node_id)
        if node:
            print(f"\nNode {node_id}:")
            print(json.dumps(node, indent=2))
        else:
            print(f"\nNode {node_id} not found")
    else:
        print("\nSample node IDs:", list(node_map.keys())[:10])

