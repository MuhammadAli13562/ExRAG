"""
Simplified markdown to tree converter.
Converts markdown files to hierarchical tree structure with text content included.
"""

import re
import os
import json


def extract_nodes_from_markdown(markdown_content):
    """Extract all markdown headers from content."""
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    
    lines = markdown_content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        # Check for code block delimiters (triple backticks)
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        
        # Skip empty lines
        if not stripped_line:
            continue
        
        # Only look for headers when not inside a code block
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list, markdown_lines):
    """Extract text content for each node from markdown lines."""
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        
        if header_match is None:
            print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
            continue
            
        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1))
        }
        all_nodes.append(processed_node)
    
    # Extract text for each node
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1 
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1 
        else:
            end_line = len(markdown_lines)
        
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
    
    return all_nodes


def build_tree_from_nodes(node_list):
    """Build hierarchical tree structure from flat list of nodes."""
    if not node_list:
        return []
    
    stack = []
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []
        }
        node_counter += 1
        
        # Pop from stack until we find a parent with lower level
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        # Add to appropriate parent or root
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes


def clean_empty_nodes(tree_nodes):
    """Remove empty 'nodes' arrays from tree structure."""
    cleaned_nodes = []
    
    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num']
        }
        
        if node['nodes']:
            cleaned_node['nodes'] = clean_empty_nodes(node['nodes'])
        
        cleaned_nodes.append(cleaned_node)
    
    return cleaned_nodes


def md_to_tree(md_path):
    """
    Convert markdown file to tree structure with text included.
    
    Args:
        md_path: Path to markdown file
        
    Returns:
        Dictionary with:
            - doc_name: Name of the markdown file (without extension)
            - structure: List of tree nodes with title, node_id, text, line_num, and nested nodes
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)
    
    print(f"Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    
    print(f"Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)
    
    print(f"Cleaning tree structure...")
    tree_structure = clean_empty_nodes(tree_structure)
    
    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'structure': tree_structure,
    }


def print_tree(tree, indent=0):
    """Print tree structure in a readable format."""
    for node in tree:
        print('  ' * indent + f"[{node['node_id']}] {node['title']}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python md_to_tree_simple.py <markdown_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    
    if not os.path.isfile(md_file):
        print(f"Error: File not found: {md_file}")
        sys.exit(1)
    
    # Convert markdown to tree
    result = md_to_tree(md_file)
    
    # Save to JSON
    output_path = f"./results/{result['doc_name']}_structure__simple.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Tree structure saved to: {output_path}")

