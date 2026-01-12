"""
RetrievalLoop sub-graph for ReAct-style seed processing.
"""
import re
import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from ..state import RetrievalLoopState
from ..constants import (
    RETRIEVAL_SCOPE_PARAMS,
    MAX_RETRIEVAL_SCOPE,
    MAX_SYNTHESIS_RETRIES,
    RETRIEVAL_MAX_ITERATIONS,
)
from ..prompts import RETRIEVAL_REFLECTION_PROMPT
from ..grading import grade_single_node, grade_neighbors_batch
from ..helpers import is_evidence_sufficient, should_early_exit, count_evidence_grades
from ...tools import RetrievalTools

logger = logging.getLogger(__name__)


def create_retrieval_loop_subgraph(
    llm: ChatOpenAI,
    retrieval_tools_instance: RetrievalTools,
    text_collection: str,
    tracer
):
    """
    Create the RetrievalLoop sub-graph for ReAct-style seed processing.

    This sub-graph implements:
    - Reflection after processing seeds (evaluate progress, decide action)
    - Adaptive expansion radius based on seed grade
    - Evidence sufficiency check before synthesis
    - Retry with expanded scope if evidence is insufficient
    - Synthesis retry with feedback if validation fails
    - Early exit when excellent evidence is found

    Args:
        llm: The LLM instance to use
        retrieval_tools_instance: RetrievalTools instance for document access
        text_collection: Name of the text-indexed collection
        tracer: Langfuse tracer for observability

    Returns:
        Compiled LangGraph sub-graph for retrieval loop
    """

    # -------------------------------------------------------------------------
    # Node Functions
    # -------------------------------------------------------------------------

    def init_loop(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Initialize the retrieval loop state.
        """
        initial_seeds = state.get("initial_seeds") or []
        
        logger.info(f"[RETRIEVAL_LOOP] init_loop: {len(initial_seeds)} initial seeds")
        
        return {
            "pending_seeds": initial_seeds,
            "visited_node_ids": [],
            "evidence_pool": [],
            "iteration": 0,
            "max_iterations": RETRIEVAL_MAX_ITERATIONS,
            "scope_level": 0,
            "status": "processing",
            "seeds_processed": 0,
            "total_seeds": len(initial_seeds),
            "reflection": "",
            "last_action": "initialized",
            "high_grade_count": 0,
            "medium_grade_count": 0,
            "final_answer": "",
            "synthesis_attempts": 0,
            "validation": {},
            "synthesis_feedback": "",
            "result": {},
        }

    def process_seed(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Process ONE seed at a time with adaptive expansion radius.
        """
        seeds = state.get("pending_seeds") or []
        if not seeds:
            logger.info("[RETRIEVAL_LOOP] process_seed: No more seeds")
            return {
                "status": "reflecting",
                "last_action": "no_seeds_remaining",
            }

        current_seed = seeds[0]
        remaining_seeds = seeds[1:]
        visited = set(state.get("visited_node_ids") or [])
        evidence_pool = list(state.get("evidence_pool") or [])
        evidence_ids = {e.get("node_id") for e in evidence_pool}

        seed_id = current_seed.get("node_id", "unknown")
        seeds_processed = state.get("seeds_processed", 0)
        scope_level = state.get("scope_level", 0)
        scope = RETRIEVAL_SCOPE_PARAMS.get(scope_level, RETRIEVAL_SCOPE_PARAMS[0])

        logger.info(f"[RETRIEVAL_LOOP] process_seed: {seed_id}, {len(remaining_seeds)} remaining, scope={scope['name']}")

        with tracer.span("retrieval_loop.process_seed", input={"seed_id": seed_id}) as span:
            # 1. Grade this seed individually
            t0 = time.time()
            seed_grade = grade_single_node(current_seed, state["user_query"], llm)
            dt = int((time.time() - t0) * 1000)

            tracer.event(
                "retrieval_loop.grade_seed",
                input={"seed_id": seed_id},
                output={"grade": seed_grade},
                metadata={"duration_ms": dt},
            )

            logger.info(f"[RETRIEVAL_LOOP] Seed {seed_id} graded as: {seed_grade}")

            new_visited = list(visited) + [seed_id]

            if seed_grade not in ("high", "medium"):
                # Seed not relevant → skip
                logger.info(f"[RETRIEVAL_LOOP] Seed {seed_id} not relevant, skipping")
                return {
                    "pending_seeds": remaining_seeds,
                    "visited_node_ids": new_visited,
                    "seeds_processed": seeds_processed + 1,
                    "status": "reflecting",
                    "last_action": f"skipped_irrelevant_seed_{seed_id}",
                    "iteration": state.get("iteration", 0) + 1,
                }

            # 2. ADAPTIVE RADIUS: Expand more around high-grade seeds
            base_radius = scope["radius"]
            if seed_grade == "high":
                radius = base_radius + 2  # Expand more around high-grade seeds
            else:
                radius = base_radius

            logger.info(f"[RETRIEVAL_LOOP] Expanding seed {seed_id} with radius={radius} (base={base_radius}, grade={seed_grade})")

            try:
                expanded = retrieval_tools_instance.expand_around(
                    node_id=seed_id,
                    collection_name=text_collection,
                    radius=radius
                )
                nodes_above = expanded.get("nodes_above", [])
                nodes_below = expanded.get("nodes_below", [])
                all_expanded_nodes = nodes_above + nodes_below
                neighbors = [n for n in all_expanded_nodes
                             if n.get("node_id") not in visited
                             and n.get("node_id") != seed_id
                             and n.get("node_id") not in evidence_ids]

                logger.info(f"[RETRIEVAL_LOOP] Expanded {seed_id}: {len(all_expanded_nodes)} total, {len(neighbors)} new")
            except Exception as e:
                logger.warning(f"[RETRIEVAL_LOOP] Failed to expand {seed_id}: {e}")
                neighbors = []

            # 3. Batch-grade neighbors
            graded_neighbors = []
            if neighbors:
                t0 = time.time()
                graded_neighbors = grade_neighbors_batch(neighbors, state["user_query"], llm)
                dt = int((time.time() - t0) * 1000)
                tracer.event(
                    "retrieval_loop.grade_neighbors",
                    input={"seed_id": seed_id, "num_neighbors": len(neighbors)},
                    output={"graded_count": len(graded_neighbors)},
                    metadata={"duration_ms": dt},
                )

            relevant_neighbors = [n for n in graded_neighbors if n.get("relevance_grade") in ("high", "medium")]

            # 4. Add to evidence pool
            seed_with_grade = {
                **current_seed,
                "relevance_grade": seed_grade,
                "seed_order": seeds_processed,
            }
            if seed_id not in evidence_ids:
                evidence_pool.append(seed_with_grade)
                evidence_ids.add(seed_id)

            for n in relevant_neighbors:
                nid = n.get("node_id")
                if nid and nid not in evidence_ids:
                    evidence_pool.append({**n, "seed_order": seeds_processed})
                    evidence_ids.add(nid)

            # Sort evidence pool
            evidence_pool.sort(key=lambda x: (
                0 if x.get("relevance_grade") == "high" else 1,
                x.get("seed_order", 999),
                -(x.get("similarity_score") or 0)
            ))

            # Update visited
            new_visited = list(visited) + [seed_id] + [n.get("node_id") for n in neighbors if n.get("node_id")]

            # Count grades
            high_count, medium_count = count_evidence_grades(evidence_pool)

            logger.info(f"[RETRIEVAL_LOOP] Seed {seed_id}: added {1 + len(relevant_neighbors)} to evidence (total: {len(evidence_pool)}, high={high_count}, medium={medium_count})")

            if span is not None and hasattr(span, "update"):
                span.update(output={
                    "action": "expanded",
                    "grade": seed_grade,
                    "radius_used": radius,
                    "neighbors_added": len(relevant_neighbors),
                    "total_evidence": len(evidence_pool),
                })

            return {
                "pending_seeds": remaining_seeds,
                "evidence_pool": evidence_pool,
                "visited_node_ids": new_visited,
                "seeds_processed": seeds_processed + 1,
                "high_grade_count": high_count,
                "medium_grade_count": medium_count,
                "status": "reflecting",
                "last_action": f"expanded_seed_{seed_id}_radius_{radius}",
                "iteration": state.get("iteration", 0) + 1,
            }

    def reflect(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Reflect on retrieval progress and decide next action.
        """
        scope_level = state.get("scope_level", 0)
        scope = RETRIEVAL_SCOPE_PARAMS.get(scope_level, RETRIEVAL_SCOPE_PARAMS[0])
        
        high_count = state.get("high_grade_count", 0)
        medium_count = state.get("medium_grade_count", 0)
        seeds_processed = state.get("seeds_processed", 0)
        total_seeds = state.get("total_seeds", 0)
        pending_seeds = state.get("pending_seeds") or []
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", RETRIEVAL_MAX_ITERATIONS)

        # Get query context for evidence thresholds
        query_type = state.get("query_type", "conceptual")
        route = state.get("route", "semantic_only")
        
        # Check for early exit condition (excellent evidence)
        # Anthropic principle: "Set clear stopping points to control costs"
        if should_early_exit(high_count, medium_count, query_type, route):
            logger.info(f"[RETRIEVAL_LOOP] reflect: Early exit - excellent evidence "
                       f"(high={high_count}, medium={medium_count}, route={route})")
            return {
                "status": "checking_evidence",
                "reflection": f"Early exit: Excellent evidence collected (high={high_count}, medium={medium_count})",
            }

        # Check if evidence is already sufficient (before asking LLM)
        # Anthropic principle: "Use explicit criteria instead of always asking LLM"
        if is_evidence_sufficient(high_count, medium_count, query_type, route):
            logger.info(f"[RETRIEVAL_LOOP] reflect: Evidence sufficient "
                       f"(high={high_count}, medium={medium_count}, route={route}), proceeding to check")
            return {
                "status": "checking_evidence",
                "reflection": f"Evidence sufficient (high={high_count}, medium={medium_count}), proceeding to synthesis",
            }

        # Special case: If we have a high-grade seed and all seeds are processed (or only 1 seed),
        # and we have at least 1 high + 2 medium, consider it sufficient for structural queries
        # This handles cases where navigator found exact match and we've expanded around it
        evidence_pool = state.get("evidence_pool") or []
        has_structural_seed = any(
            e.get("source") == "structural" and e.get("relevance_grade") == "high"
            for e in evidence_pool
        )
        if has_structural_seed and high_count >= 1 and medium_count >= 2 and (not pending_seeds or total_seeds == 1):
            logger.info(f"[RETRIEVAL_LOOP] reflect: High-grade structural match found with sufficient context (high={high_count}, medium={medium_count}), proceeding to check")
            return {
                "status": "checking_evidence",
                "reflection": f"High-grade structural match with context (high={high_count}, medium={medium_count}), proceeding to synthesis",
            }

        # Check if all seeds processed
        if not pending_seeds:
            logger.info(f"[RETRIEVAL_LOOP] reflect: All seeds processed, checking evidence")
            return {
                "status": "checking_evidence",
                "reflection": f"All {seeds_processed} seeds processed. Evidence: high={high_count}, medium={medium_count}",
            }

        # Check iteration limit
        if iteration >= max_iterations:
            logger.info(f"[RETRIEVAL_LOOP] reflect: Max iterations reached ({iteration}/{max_iterations})")
            return {
                "status": "checking_evidence",
                "reflection": f"Max iterations reached. Evidence: high={high_count}, medium={medium_count}",
            }

        # Use LLM for reflection
        prompt = RETRIEVAL_REFLECTION_PROMPT.format(
            query=state.get("user_query", ""),
            processed=seeds_processed,
            total=total_seeds,
            evidence_count=len(state.get("evidence_pool") or []),
            high_count=high_count,
            medium_count=medium_count,
            last_action=state.get("last_action", "None"),
            scope_name=scope["name"],
            iteration=iteration,
            max_iterations=max_iterations,
        )

        t0 = time.time()
        response = llm.invoke([
            SystemMessage(content="You are evaluating retrieval progress. Be concise."),
            HumanMessage(content=prompt)
        ])
        dt = int((time.time() - t0) * 1000)

        raw = response.content if hasattr(response, 'content') else str(response)

        # Parse reflection: sufficient|action|reason
        parts = raw.strip().split("|")
        sufficient = parts[0].strip().lower() if len(parts) > 0 else "uncertain"
        action = parts[1].strip().lower() if len(parts) > 1 else "continue"
        reason = parts[2].strip() if len(parts) > 2 else ""

        logger.info(f"[RETRIEVAL_LOOP] reflect: sufficient={sufficient}, action={action}, reason={reason}")

        tracer.event(
            "retrieval_loop.reflect",
            input={"iteration": iteration, "evidence": len(state.get("evidence_pool") or [])},
            output={"sufficient": sufficient, "action": action, "reason": reason},
            metadata={"duration_ms": dt},
        )

        # Determine new status
        if action == "early_exit" or sufficient == "yes":
            new_status = "checking_evidence"
        elif action == "expand_scope":
            if scope_level < MAX_RETRIEVAL_SCOPE:
                new_status = "retrying_search"
            else:
                new_status = "checking_evidence"  # Can't expand further
        elif action == "give_up":
            new_status = "checking_evidence"  # Proceed with what we have
        else:
            new_status = "processing"  # Continue processing seeds

        return {
            "status": new_status,
            "reflection": f"{sufficient}: {reason}",
        }

    def check_evidence(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Check if evidence is sufficient for synthesis.
        
        Anthropic principle: Use explicit thresholds based on query type.
        """
        high_count = state.get("high_grade_count", 0)
        medium_count = state.get("medium_grade_count", 0)
        scope_level = state.get("scope_level", 0)
        evidence_pool = state.get("evidence_pool") or []
        query_type = state.get("query_type", "conceptual")
        route = state.get("route", "semantic_only")

        sufficient = is_evidence_sufficient(high_count, medium_count, query_type, route)

        # Special case: High-grade structural seed with context is sufficient
        # This handles cases where navigator found exact match (e.g., "chapter 4 review questions")
        has_structural_seed = any(
            e.get("source") == "structural" and e.get("relevance_grade") == "high"
            for e in evidence_pool
        )
        if not sufficient and has_structural_seed and high_count >= 1 and medium_count >= 2:
            logger.info(f"[RETRIEVAL_LOOP] check_evidence: High-grade structural match with context (high={high_count}, medium={medium_count}), considering sufficient")
            sufficient = True

        logger.info(f"[RETRIEVAL_LOOP] check_evidence: high={high_count}, medium={medium_count}, "
                   f"route={route}, sufficient={sufficient}")

        if sufficient or scope_level >= MAX_RETRIEVAL_SCOPE:
            # Proceed to synthesis (either sufficient or max scope reached)
            if not sufficient:
                logger.warning(f"[RETRIEVAL_LOOP] Evidence insufficient but max scope reached, proceeding anyway")
            return {
                "status": "synthesizing",
                "reflection": f"Evidence check: high={high_count}, medium={medium_count}, proceeding to synthesis",
            }
        else:
            # Retry with broader scope
            logger.info(f"[RETRIEVAL_LOOP] Evidence insufficient, retrying with broader scope")
            return {
                "status": "retrying_search",
                "reflection": f"Evidence insufficient (high={high_count}, medium={medium_count}), expanding scope",
            }

    def retry_search(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Retry search with expanded scope parameters.
        """
        current_scope = state.get("scope_level", 0)
        new_scope = min(current_scope + 1, MAX_RETRIEVAL_SCOPE)
        scope_params = RETRIEVAL_SCOPE_PARAMS.get(new_scope, RETRIEVAL_SCOPE_PARAMS[MAX_RETRIEVAL_SCOPE])

        logger.info(f"[RETRIEVAL_LOOP] retry_search: Expanding from scope {current_scope} to {new_scope} ({scope_params['name']})")

        # Get new seeds with expanded parameters
        subqueries = state.get("subqueries") or [state.get("user_query", "")]
        visited = set(state.get("visited_node_ids") or [])

        all_new_candidates: Dict[str, Dict[str, Any]] = {}
        for query in subqueries:
            try:
                results = retrieval_tools_instance.search_by_text(
                    query, state["text_collection"], top_k=scope_params["top_k"]
                )
                for r in results:
                    node_id = r.get("node_id")
                    if node_id and node_id not in visited and node_id not in all_new_candidates:
                        r["source"] = "retry_semantic"
                        all_new_candidates[node_id] = r
            except Exception as e:
                logger.warning(f"[RETRIEVAL_LOOP] retry_search failed for query '{query}': {e}")

        # Sort by similarity and take top max_seeds
        sorted_candidates = sorted(
            all_new_candidates.values(),
            key=lambda x: -(x.get("similarity_score") or 0)
        )
        new_seeds = sorted_candidates[:scope_params["max_seeds"]]

        logger.info(f"[RETRIEVAL_LOOP] retry_search: Found {len(new_seeds)} new seeds at {scope_params['name']} scope")

        tracer.event(
            "retrieval_loop.retry_search",
            input={"old_scope": current_scope, "new_scope": new_scope},
            output={"new_seeds": len(new_seeds)},
        )

        return {
            "pending_seeds": new_seeds,
            "scope_level": new_scope,
            "total_seeds": len(new_seeds),
            "seeds_processed": 0,  # Reset for new scope
            "iteration": 0,  # Reset iteration count
            "status": "processing" if new_seeds else "synthesizing",
            "last_action": f"expanded_to_{scope_params['name']}_scope",
            "reflection": f"Retrying with {scope_params['name']} scope (top_k={scope_params['top_k']}, radius={scope_params['radius']})",
        }

    def synthesize(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Synthesize final answer from evidence pool.
        """
        user_query = state["user_query"]
        evidence_pool = state.get("evidence_pool") or []
        synthesis_attempts = state.get("synthesis_attempts", 0)
        synthesis_feedback = state.get("synthesis_feedback", "")

        # Take top evidence nodes
        high_nodes = [n for n in evidence_pool if n.get("relevance_grade") == "high"]
        medium_nodes = [n for n in evidence_pool if n.get("relevance_grade") == "medium"]

        max_high = 8
        max_medium = 4
        evidence_nodes = high_nodes[:max_high] + medium_nodes[:max_medium]

        logger.info(f"[RETRIEVAL_LOOP] synthesize: attempt {synthesis_attempts + 1}, using {len(evidence_nodes)} evidence nodes")

        with tracer.span("retrieval_loop.synthesize", input={"num_evidence": len(evidence_nodes), "attempt": synthesis_attempts + 1}) as span:
            evidence_lines: List[str] = []
            for n in evidence_nodes:
                nid = n.get("node_id")
                title = (n.get("title") or "").strip()
                text = (n.get("text") or "").strip()
                grade = n.get("relevance_grade", "unknown")

                # Full text for high-grade, 5000 for medium
                if grade == "high":
                    snippet = text
                else:
                    snippet = text[:5000]

                evidence_lines.append(f"[Node {nid}: {title}] (grade: {grade})\n{snippet}")

            # Build system message with feedback if retrying
            sys_content = (
                "You are a retrieval QA system.\n"
                "You MUST cite every sentence with [Node XXXX] or [Node XXXX: Title].\n"
                "Only use information present in the provided evidence nodes.\n"
                "If evidence is insufficient, say what is missing and still cite the closest relevant nodes.\n"
            )
            
            if synthesis_feedback:
                sys_content += f"\n{synthesis_feedback}"

            sys = SystemMessage(content=sys_content)
            human = HumanMessage(
                content=("QUESTION:\n" + user_query + "\n\nEVIDENCE:\n" + "\n\n---\n\n".join(evidence_lines))
            )

            t0 = time.time()
            resp = llm.invoke([sys, human])
            dt = int((time.time() - t0) * 1000)
            answer = resp.content if isinstance(resp, AIMessage) else str(resp)

            tracer.event(
                "retrieval_loop.synthesize_llm",
                input={"attempt": synthesis_attempts + 1},
                output={"answer_length": len(answer)},
                metadata={"duration_ms": dt},
            )

            if span is not None and hasattr(span, "update"):
                span.update(output={"answer_length": len(answer)})

            return {
                "final_answer": answer,
                "synthesis_attempts": synthesis_attempts + 1,
                "status": "validating",
                "last_action": f"synthesized_attempt_{synthesis_attempts + 1}",
            }

    def validate(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Validate citations and determine next action.
        """
        answer = state.get("final_answer") or ""
        evidence_pool = state.get("evidence_pool") or []
        evidence_ids = {e.get("node_id") for e in evidence_pool if e.get("node_id")}
        synthesis_attempts = state.get("synthesis_attempts", 0)

        # Parse sentences and check citations
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        citation_re = re.compile(r"\[Node\s+(\d{4})(?::[^\]]+)?\]")

        missing_citation: List[str] = []
        cited_ids: List[str] = []

        for sent in sentences:
            if len(re.sub(r"\W+", "", sent)) < 15:
                continue
            found = citation_re.findall(sent)
            if not found:
                missing_citation.append(sent)
            else:
                cited_ids.extend(found)

        invalid_ids = sorted({cid for cid in cited_ids if cid not in evidence_ids})
        
        validation = {
            "ok": (len(missing_citation) == 0) and (len(invalid_ids) == 0) and (len(sentences) > 0),
            "num_sentences": len(sentences),
            "missing_citation_count": len(missing_citation),
            "invalid_citation_node_ids": invalid_ids,
            "cited_ids": cited_ids[:20],
        }

        logger.info(f"[RETRIEVAL_LOOP] validate: ok={validation['ok']}, missing={len(missing_citation)}, invalid={len(invalid_ids)}")

        tracer.event(
            "retrieval_loop.validate",
            input={"attempt": synthesis_attempts},
            output=validation,
        )

        # Determine status based on validation
        if validation["ok"]:
            new_status = "success"
        elif synthesis_attempts >= MAX_SYNTHESIS_RETRIES:
            # Max retries reached, return best effort
            logger.warning(f"[RETRIEVAL_LOOP] Max synthesis retries ({MAX_SYNTHESIS_RETRIES}) reached, returning best effort")
            new_status = "best_effort"
        elif len(missing_citation) > len(sentences) // 2:
            # More than half sentences missing citations - retry
            new_status = "retrying_synthesis"
        else:
            # Acceptable - some issues but usable
            new_status = "best_effort"

        return {
            "validation": validation,
            "status": new_status,
        }

    def retry_synthesis(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Retry synthesis with feedback about previous errors.
        """
        validation = state.get("validation", {})
        missing = validation.get("missing_citation_count", 0)
        invalid = validation.get("invalid_citation_node_ids", [])

        feedback = f"""IMPORTANT CORRECTIONS NEEDED:
Previous answer had {missing} sentences without citations and {len(invalid)} invalid node citations.

STRICT REQUIREMENTS:
1. Every factual sentence MUST have [Node XXXX] citation
2. Only use node IDs from the evidence provided below
3. If information is incomplete, explicitly state this with the closest citation
4. Invalid node IDs from previous attempt: {invalid} - DO NOT use these"""

        logger.info(f"[RETRIEVAL_LOOP] retry_synthesis: Adding feedback for retry")

        return {
            "synthesis_feedback": feedback,
            "status": "synthesizing",
            "last_action": "preparing_synthesis_retry",
        }

    def exit_success(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Exit with successful answer.
        """
        logger.info(f"[RETRIEVAL_LOOP] exit_success: Returning validated answer")
        
        return {
            "result": {
                "final_answer": state.get("final_answer", ""),
                "evidence_pool": state.get("evidence_pool", []),
                "validation": state.get("validation", {}),
                "status": "success",
                "synthesis_attempts": state.get("synthesis_attempts", 0),
                "scope_level": state.get("scope_level", 0),
            }
        }

    def exit_best_effort(state: RetrievalLoopState) -> Dict[str, Any]:
        """
        Exit with best available answer (may have some issues).
        """
        logger.info(f"[RETRIEVAL_LOOP] exit_best_effort: Returning best available answer")
        
        return {
            "result": {
                "final_answer": state.get("final_answer", ""),
                "evidence_pool": state.get("evidence_pool", []),
                "validation": state.get("validation", {}),
                "status": "best_effort",
                "synthesis_attempts": state.get("synthesis_attempts", 0),
                "scope_level": state.get("scope_level", 0),
            }
        }

    # -------------------------------------------------------------------------
    # Routing Function
    # -------------------------------------------------------------------------

    def route_retrieval_loop(state: RetrievalLoopState) -> str:
        """Route based on retrieval loop status."""
        status = state.get("status", "processing")

        logger.debug(f"[RETRIEVAL_LOOP] route: status={status}")

        if status == "success":
            return "exit_success"
        if status == "best_effort":
            return "exit_best_effort"
        if status == "processing":
            return "process_seed"
        if status == "reflecting":
            return "reflect"
        if status == "checking_evidence":
            return "check_evidence"
        if status == "retrying_search":
            return "retry_search"
        if status == "synthesizing":
            return "synthesize"
        if status == "validating":
            return "validate"
        if status == "retrying_synthesis":
            return "retry_synthesis"

        # Default fallback
        return "process_seed"

    # -------------------------------------------------------------------------
    # Build Sub-graph
    # -------------------------------------------------------------------------

    retrieval_loop = StateGraph(RetrievalLoopState)

    # Add nodes
    retrieval_loop.add_node("init_loop", init_loop)
    retrieval_loop.add_node("process_seed", process_seed)
    retrieval_loop.add_node("reflect", reflect)
    retrieval_loop.add_node("check_evidence", check_evidence)
    retrieval_loop.add_node("retry_search", retry_search)
    retrieval_loop.add_node("synthesize", synthesize)
    retrieval_loop.add_node("validate", validate)
    retrieval_loop.add_node("retry_synthesis", retry_synthesis)
    retrieval_loop.add_node("exit_success", exit_success)
    retrieval_loop.add_node("exit_best_effort", exit_best_effort)

    # Entry point
    retrieval_loop.set_entry_point("init_loop")

    # Fixed edge from init to first route
    retrieval_loop.add_edge("init_loop", "process_seed")

    # Conditional routing from all states
    retrieval_loop.add_conditional_edges(
        "process_seed",
        route_retrieval_loop,
        {
            "reflect": "reflect",
            "process_seed": "process_seed",
            "check_evidence": "check_evidence",
        }
    )

    retrieval_loop.add_conditional_edges(
        "reflect",
        route_retrieval_loop,
        {
            "processing": "process_seed",
            "process_seed": "process_seed",
            "checking_evidence": "check_evidence",
            "check_evidence": "check_evidence",
            "retrying_search": "retry_search",
            "retry_search": "retry_search",
        }
    )

    retrieval_loop.add_conditional_edges(
        "check_evidence",
        route_retrieval_loop,
        {
            "synthesizing": "synthesize",
            "synthesize": "synthesize",
            "retrying_search": "retry_search",
            "retry_search": "retry_search",
        }
    )

    retrieval_loop.add_conditional_edges(
        "retry_search",
        route_retrieval_loop,
        {
            "processing": "process_seed",
            "process_seed": "process_seed",
            "synthesizing": "synthesize",
            "synthesize": "synthesize",
        }
    )

    retrieval_loop.add_edge("synthesize", "validate")

    retrieval_loop.add_conditional_edges(
        "validate",
        route_retrieval_loop,
        {
            "success": "exit_success",
            "exit_success": "exit_success",
            "best_effort": "exit_best_effort",
            "exit_best_effort": "exit_best_effort",
            "retrying_synthesis": "retry_synthesis",
            "retry_synthesis": "retry_synthesis",
        }
    )

    retrieval_loop.add_edge("retry_synthesis", "synthesize")

    # Terminal edges
    retrieval_loop.add_edge("exit_success", END)
    retrieval_loop.add_edge("exit_best_effort", END)

    return retrieval_loop.compile()
