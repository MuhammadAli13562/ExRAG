"""
Planner node for the retrieval agent.

The planner analyzes the user query using an LLM to:
1. Detect structural intent (chapter refs, section keywords, position hints)
2. Decide routing: structural navigation vs semantic search
3. Set retrieval parameters based on query complexity
"""
import json
import re
import time
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..state import AgentState

logger = logging.getLogger(__name__)

# System prompt for structural intent detection
STRUCTURAL_DETECTION_PROMPT = """You are analyzing a user query to detect structural intent for document retrieval.

Analyze the query and return ONLY valid JSON with these keys:

{
  "has_structural": boolean,  // true if query references document structure
  "structural_hints": {
    "chapter": integer or null,      // chapter number (e.g., 4 for "chapter 4", "chapter four", "ch 4")
    "section": string or null,       // section identifier (e.g., "4.2", "3" for "section 4.2")
    "part": integer or null,         // part number (e.g., 2 for "part 2", "part II")
    "appendix": string or null,      // appendix identifier (e.g., "A", "3")
    "page": integer or null,         // page number
    "figure": string or null,        // figure identifier (e.g., "3.1")
    "table": string or null,         // table identifier (e.g., "4.2")
    "position": string or null,      // "beginning", "middle", "end", or null
    "section_keywords": array        // keywords like ["review", "summary", "exercises", "key terms"]
  },
  "query_type": string,              // "structural", "conceptual", or "hybrid"
  "search_query": string             // the core semantic query to search (strip structural refs)
}

Guidelines:
- "chapter four" = chapter 4, "ch. 12" = chapter 12, "chapter twelve" = chapter 12
- "part II" = part 2, "part two" = part 2
- Position: "end of chapter", "last section" = "end"; "beginning", "start", "intro" = "beginning"
- Section keywords: review, questions, summary, exercises, key terms, vocabulary, glossary,
  introduction, conclusion, case study, examples, learning objectives, quiz, practice, etc.
- query_type: "structural" if query is primarily about locating a section, "conceptual" if about content
- search_query: extract the conceptual part for semantic search (e.g., "what is mitosis" from
  "explain mitosis from chapter 4")

Examples:
Query: "What are the review questions at the end of chapter 4?"
→ has_structural: true, chapter: 4, position: "end", section_keywords: ["review", "questions"]
   query_type: "structural", search_query: "review questions"

Query: "Explain the key concepts of photosynthesis"
→ has_structural: false, query_type: "conceptual", search_query: "key concepts of photosynthesis"

Query: "Summary of chapter 3 section 2"
→ has_structural: true, chapter: 3, section: "2", section_keywords: ["summary"]
   query_type: "structural", search_query: "summary"
"""


def create_planner(llm: ChatOpenAI, tracer, agent_model: str):
    """
    Create the planner node function.

    Args:
        llm: The LLM instance to use
        tracer: Langfuse tracer for observability
        agent_model: Model name for logging

    Returns:
        Planner node function
    """

    def _state_summary(state: AgentState) -> Dict[str, Any]:
        """Compact state summary for tracing (avoids dumping huge node texts)."""
        plan = state.get("plan") or {}
        return {
            "user_query": state.get("user_query"),
            "plan": plan,
            "pending_seeds": len(state.get("pending_seeds") or []),
            "num_visited": len(state.get("visited_node_ids") or []),
            "num_evidence": len(state.get("evidence_pool") or []),
            "processing_complete": state.get("processing_complete", False),
            "validation": state.get("validation") or {},
        }

    def _clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
        try:
            n = int(val)
        except Exception:
            return default
        return max(lo, min(hi, n))

    def _extract_json(raw: str) -> str:
        """Extract JSON from LLM response that may be wrapped in markdown code blocks."""
        # Try to extract from ```json ... ``` or ``` ... ``` blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Try to find a JSON object directly (starts with { and ends with })
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json_match.group(0).strip()
        
        # Return as-is if no patterns match
        return raw.strip()

    def _parse_llm_response(raw: str, user_query: str) -> Dict[str, Any]:
        """Parse LLM response with fallback handling."""
        # First try to extract JSON from markdown code blocks or find JSON object
        extracted = _extract_json(raw)
        
        try:
            parsed = json.loads(extracted)
        except Exception as e:
            # Fallback if LLM doesn't return valid JSON
            logger.warning(f"[PLANNER] Failed to parse LLM response as JSON: {e}")
            logger.debug(f"[PLANNER] Raw response was: {raw[:500]}")
            return {
                "has_structural": False,
                "structural_hints": {
                    "chapter": None,
                    "section": None,
                    "part": None,
                    "appendix": None,
                    "page": None,
                    "figure": None,
                    "table": None,
                    "position": None,
                    "section_keywords": [],
                },
                "query_type": "conceptual",
                "search_query": user_query,
            }

        # Normalize structural_hints
        hints = parsed.get("structural_hints", {})
        normalized_hints = {
            "chapter": hints.get("chapter"),
            "section": hints.get("section"),
            "part": hints.get("part"),
            "appendix": hints.get("appendix"),
            "page": hints.get("page"),
            "figure": hints.get("figure"),
            "table": hints.get("table"),
            "position": hints.get("position"),
            "section_keywords": hints.get("section_keywords", []),
            "has_structural": parsed.get("has_structural", False),
        }

        return {
            "has_structural": parsed.get("has_structural", False),
            "structural_hints": normalized_hints,
            "query_type": parsed.get("query_type", "conceptual"),
            "search_query": parsed.get("search_query", user_query),
        }

    def planner(state: AgentState) -> Dict[str, Any]:
        """
        Analyze query and plan retrieval strategy.

        Returns a dict in state['plan'] with keys:
          - has_structural: bool - whether to use navigator
          - structural_hints: dict - hints for navigator (chapter, section, etc.)
          - query_type: str - "structural", "conceptual", or "hybrid"
          - search_query: str - query for semantic search
          - subqueries: list[str] - just the original query (no LLM subquery generation)
          - top_k: int
          - radius: int
          - max_evidence_nodes: int
          - max_seeds: int
        """
        user_query = state["user_query"]
        logger.info("[STAGE] planner: analyzing query and building retrieval plan")

        with tracer.span("planner", input={"state": _state_summary(state)}) as span:
            sys = SystemMessage(content=STRUCTURAL_DETECTION_PROMPT)

            t0 = time.time()
            resp = llm.invoke([sys, HumanMessage(content=user_query)])
            dt = int((time.time() - t0) * 1000)
            raw = resp.content if isinstance(resp, AIMessage) else str(resp)

            tracer.event(
                "llm.planner",
                input={"messages": [sys.content, user_query]},
                output={"response": raw},
                metadata={"duration_ms": dt, "model": agent_model},
            )

            # Parse LLM response
            parsed = _parse_llm_response(raw, user_query)

            # Build plan with retrieval parameters
            # ROUTING DECISION (Anthropic best practice: route to appropriate handler)
            has_structural = parsed["has_structural"]
            hints = parsed["structural_hints"]
            query_type = parsed["query_type"]
            
            # Determine route based on structural completeness
            if has_structural and hints.get("chapter") is not None:
                if query_type == "structural":
                    # Clear structural query: "review questions chapter 4"
                    # Use deterministic navigation (no LLM loop needed)
                    route = "structural_only"
                else:
                    # Hybrid: "explain mitosis from chapter 4"
                    # Use both structural nav + semantic search
                    route = "hybrid"
            elif has_structural and hints.get("section_keywords"):
                # Has keywords but no chapter - still try structural
                route = "hybrid"
            else:
                # No clear structure: "what is photosynthesis"
                # Skip navigator entirely, use semantic search only
                route = "semantic_only"

            plan_norm = {
                # Routing decision (NEW)
                "route": route,
                
                # Structural detection results
                "has_structural": parsed["has_structural"],
                "structural_hints": parsed["structural_hints"],
                "query_type": parsed["query_type"],
                "search_query": parsed["search_query"],

                # For semantic search - just use the search_query, no multiple subqueries
                "subqueries": [parsed["search_query"]],

                # Retrieval parameters (reasonable defaults)
                "top_k": 8,
                "radius": 3,
                "max_evidence_nodes": 60,
                "max_seeds": 5,
            }

            logger.info(f"[PLANNER] Query type: {parsed['query_type']}, "
                       f"has_structural: {parsed['has_structural']}, "
                       f"route: {route}, "
                       f"search_query: '{parsed['search_query']}'")

            if parsed["has_structural"]:
                hints = parsed["structural_hints"]
                logger.info(f"[PLANNER] Structural hints: chapter={hints.get('chapter')}, "
                           f"section={hints.get('section')}, position={hints.get('position')}, "
                           f"keywords={hints.get('section_keywords')}")

            out = {
                "plan": plan_norm,
                "validation": {
                    "planner_raw_length": len(raw),
                    "has_structural": parsed["has_structural"],
                    "query_type": parsed["query_type"],
                    "route": route,
                },
            }

            if span is not None and hasattr(span, "update"):
                span.update(output=out)

            return out

    return planner
