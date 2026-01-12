"""
Constants for the retrieval agent and its sub-graphs.
"""
from typing import Dict, Any

# =========================================================================
# Navigator Sub-graph Constants
# =========================================================================

# Scope levels for navigator retry mechanism
SCOPE_PARAMS: Dict[int, Dict[str, Any]] = {
    0: {"radius": 10, "top_k": 5, "name": "narrow"},    # Initial focused search
    1: {"radius": 15, "top_k": 8, "name": "medium"},    # After first failure
    2: {"radius": 25, "top_k": 12, "name": "broad"},    # Maximum expansion
}

MAX_SCOPE_LEVEL = 2
NAVIGATOR_MAX_ITERATIONS = 8  # Per scope level


# =========================================================================
# RetrievalLoop Sub-graph Constants
# =========================================================================

# Scope levels for retrieval loop retry mechanism
RETRIEVAL_SCOPE_PARAMS: Dict[int, Dict[str, Any]] = {
    0: {"top_k": 8, "radius": 3, "max_seeds": 5, "name": "focused"},      # Initial focused search
    1: {"top_k": 12, "radius": 5, "max_seeds": 7, "name": "expanded"},    # After first retry
    2: {"top_k": 16, "radius": 7, "max_seeds": 10, "name": "broad"},      # Maximum expansion
}

MAX_RETRIEVAL_SCOPE = 2
MAX_SYNTHESIS_RETRIES = 2
RETRIEVAL_MAX_ITERATIONS = 15  # Max seed processing iterations per scope
