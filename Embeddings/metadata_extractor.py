"""
Extract structured metadata from JSON nodes for enhanced filtering and retrieval.
"""
import re
from typing import Dict, Any, Optional, List
from config import PAGE_PATTERN, CHAPTER_PATTERN, SECTION_PATTERN, ANCHOR_KEYWORDS


def extract_page_numbers(text: str) -> List[int]:
    """Extract page numbers from text (e.g., 'p.239')."""
    matches = re.findall(PAGE_PATTERN, text)
    return [int(m) for m in matches] if matches else []


def extract_chapter(title: str) -> Optional[str]:
    """Extract chapter number from title (e.g., '7' from '7 Electromagnetic Induction')."""
    match = re.match(CHAPTER_PATTERN, title.strip())
    return match.group(1) if match else None


def extract_section(title: str) -> Optional[str]:
    """Extract section number from title (e.g., '7.1' from '7.1 Induced e.m.f.')."""
    match = re.match(SECTION_PATTERN, title.strip())
    return match.group(1) if match else None


def detect_anchor_type(text: str, title: str) -> Optional[str]:
    """Detect if node is a special anchor (Experiment, Discussion, etc.)."""
    combined = f"{title} {text}"
    for keyword in ANCHOR_KEYWORDS:
        if keyword.lower() in combined.lower():
            return keyword
    return None


def extract_heading_level(text: str, title: str) -> int:
    """Estimate heading level from markdown markers."""
    combined = f"{title}\n{text}"
    
    # Count leading # characters
    match = re.match(r'^(#{1,6})\s', combined, re.MULTILINE)
    if match:
        return len(match.group(1))
    
    # Fallback: use section depth
    section = extract_section(title)
    if section:
        return section.count('.') + 1
    
    chapter = extract_chapter(title)
    if chapter:
        return 1
    
    return 0  # Unknown/unstructured


def count_tokens_estimate(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def extract_metadata(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all metadata from a node for ChromaDB storage.
    
    Args:
        node: Dictionary with keys: node_id, title, text, line_num, summary
        
    Returns:
        Dictionary of metadata fields
    """
    title = node.get("title", "")
    text = node.get("text", "")
    
    metadata = {
        "node_id": node.get("node_id", ""),
        "title": title,
        "line_num": node.get("line_num", 0),
        "summary": node.get("summary", ""),
        
        # Extracted fields
        "chapter": extract_chapter(title) or "",
        "section": extract_section(title) or "",
        "anchor_type": detect_anchor_type(text, title) or "",
        "heading_level": extract_heading_level(text, title),
        "token_count": count_tokens_estimate(text),
    }
    
    # Page numbers (store first one as primary, all as JSON array string)
    page_nums = extract_page_numbers(f"{title} {text}")
    if page_nums:
        metadata["page_num"] = page_nums[0]
        metadata["all_pages"] = ",".join(map(str, page_nums))
    else:
        metadata["page_num"] = 0
        metadata["all_pages"] = ""
    
    return metadata


def is_valid_node(node: Dict[str, Any]) -> bool:
    """Check if node has sufficient content to embed."""
    text = node.get("text", "").strip()
    
    # Skip empty or trivial nodes
    if len(text) < 10:
        return False
    
    # Skip nodes that are just heading markers
    if re.match(r'^#{1,6}\s*\d*\s*$', text):
        return False
    
    return True

