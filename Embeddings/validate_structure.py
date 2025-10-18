#!/usr/bin/env python3
"""
Validate JSON structure before indexing.

Checks that the JSON file has the expected format and reports statistics.
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from .metadata_extractor import extract_metadata, is_valid_node


def validate_json_file(json_path: Path):
    """Validate and analyze JSON structure."""
    print(f"Validating: {json_path.name}\n")
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract nodes
    if isinstance(data, list):
        nodes = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['structure', 'nodes', 'data', 'items']:
            if key in data:
                nodes = data[key]
                break
        else:
            print("ERROR: Unexpected JSON structure")
            print("Expected: array of nodes or object with 'structure'/'nodes' key")
            return False
    else:
        print("ERROR: Unexpected JSON structure")
        return False
    
    print(f"✓ Found {len(nodes)} total nodes\n")
    
    # Analyze structure
    valid_count = 0
    invalid_count = 0
    
    chapters = set()
    sections = set()
    anchor_types = Counter()
    heading_levels = Counter()
    token_counts = []
    
    for node in nodes:
        if is_valid_node(node):
            valid_count += 1
            metadata = extract_metadata(node)
            
            if metadata['chapter']:
                chapters.add(metadata['chapter'])
            if metadata['section']:
                sections.add(metadata['section'])
            if metadata['anchor_type']:
                anchor_types[metadata['anchor_type']] += 1
            
            heading_levels[metadata['heading_level']] += 1
            token_counts.append(metadata['token_count'])
        else:
            invalid_count += 1
    
    # Report statistics
    print(f"Valid nodes: {valid_count}")
    print(f"Invalid/empty nodes: {invalid_count}")
    print()
    
    print(f"Chapters found: {len(chapters)}")
    if chapters:
        print(f"  {sorted(chapters, key=lambda x: float(x) if x.replace('.','').isdigit() else 0)}")
    print()
    
    print(f"Sections found: {len(sections)}")
    if len(sections) <= 20:
        print(f"  {sorted(sections, key=lambda x: [int(n) for n in x.split('.')])[:20]}")
    print()
    
    if anchor_types:
        print("Anchor types:")
        for anchor, count in anchor_types.most_common():
            print(f"  {anchor}: {count}")
        print()
    
    print("Heading levels:")
    for level in sorted(heading_levels.keys()):
        print(f"  Level {level}: {heading_levels[level]}")
    print()
    
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Token statistics:")
        print(f"  Average: {avg_tokens:.1f}")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    
    print()
    print("✓ Structure validation complete!")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_structure.py <json_file>")
        return 1
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return 1
    
    try:
        success = validate_json_file(json_path)
        return 0 if success else 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

