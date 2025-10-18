"""
Unified tree building pipeline interface.
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Literal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tree.builder import md_to_tree_data, calculate_tree_stats
from common.models import TreeBuildResult
from common.settings import get_settings
from common.logging_config import get_logger

logger = get_logger("exrag.tree")


def compute_content_hash(md_path: Path) -> str:
    """Compute hash of markdown content for caching."""
    with open(md_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def build_tree(
    md_path: Path,
    output_path: Optional[Path] = None,
    mode: Literal["simple", "full"] = "full",
    force: bool = False,
) -> TreeBuildResult:
    """
    Build tree structure from markdown file.
    
    This is the main entry point for the tree building pipeline. It:
    - Parses markdown headers into a hierarchical tree
    - Extracts text content for each node
    - Writes the result to JSON
    - Returns structured metadata about the build
    
    Args:
        md_path: Path to input markdown file
        output_path: Path for output JSON (auto-generated if None)
        mode: Build mode - "simple" or "full"
        force: If True, rebuild even if output exists
    
    Returns:
        TreeBuildResult with metadata about the build
    
    Raises:
        FileNotFoundError: If md_path doesn't exist
        ValueError: If markdown parsing fails
    """
    md_path = Path(md_path)
    
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        settings = get_settings()
        suffix = "__simple" if mode == "simple" else ""
        output_path = settings.results_dir / f"{md_path.stem}_structure{suffix}.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we can skip (caching)
    content_hash = compute_content_hash(md_path)
    skipped = False
    
    if output_path.exists() and not force:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if existing.get('_content_hash') == content_hash:
                    logger.info(f"Skipping {md_path.name} (unchanged, hash={content_hash})")
                    # Load existing stats
                    structure = existing.get('structure', [])
                    total_nodes, root_nodes, max_depth = calculate_tree_stats(structure)
                    return TreeBuildResult(
                        doc_name=md_path.stem,
                        output_path=output_path,
                        total_nodes=total_nodes,
                        root_nodes=root_nodes,
                        max_depth=max_depth,
                        content_hash=content_hash,
                        mode=mode,
                        skipped=True,
                    )
        except Exception as e:
            logger.warning(f"Could not read existing output: {e}, rebuilding")
    
    # Build the tree
    logger.info(f"Building tree from {md_path.name} (mode={mode})")
    result = md_to_tree_data(md_path)
    
    # Add metadata
    result['_content_hash'] = content_hash
    result['_mode'] = mode
    
    # Calculate stats
    total_nodes, root_nodes, max_depth = calculate_tree_stats(result['structure'])
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Tree saved to {output_path} ({total_nodes} nodes, depth={max_depth})")
    
    return TreeBuildResult(
        doc_name=result['doc_name'],
        output_path=output_path,
        total_nodes=total_nodes,
        root_nodes=root_nodes,
        max_depth=max_depth,
        content_hash=content_hash,
        mode=mode,
        skipped=skipped,
    )

