"""
Tree building package for ExRAG.
Converts markdown files to hierarchical tree structures.
"""
from .pipeline import build_tree
from .builder import md_to_tree_data

__all__ = ["build_tree", "md_to_tree_data"]

