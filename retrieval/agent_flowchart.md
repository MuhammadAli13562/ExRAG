# ExRAG Retrieval Agent Architecture

## Overview

This document visualizes the **Anthropic-style agentic retrieval system** with:
1. **Routing-based Planner**: Routes queries to appropriate handlers
2. **Deterministic Navigation**: Fast structural navigation without LLM loops
3. **Navigator Sub-graph**: LLM-based navigation for complex cases
4. **RetrievalLoop Sub-graph**: ReAct-style seed processing with reflection

---

## 🎯 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ExRAG RETRIEVAL AGENT                                  │
│                    (Anthropic Best Practices Implementation)                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────┐                                                                   │
│   │ User Query  │                                                                   │
│   └──────┬──────┘                                                                   │
│          │                                                                          │
│          ▼                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           PLANNER (with Routing)                            │   │
│   │  • Detect structural intent (chapter, position, keywords)                   │   │
│   │  • Classify query type: structural | conceptual | hybrid                    │   │
│   │  • Route to: structural_only | semantic_only | hybrid                       │   │
│   └─────────────────────────────────┬───────────────────────────────────────────┘   │
│                                     │                                               │
│          ┌──────────────────────────┼──────────────────────────┐                    │
│          │                          │                          │                    │
│          ▼                          ▼                          ▼                    │
│   ┌─────────────┐          ┌─────────────────┐          ┌─────────────┐             │
│   │ STRUCTURAL  │          │     HYBRID      │          │  SEMANTIC   │             │
│   │    ONLY     │          │                 │          │    ONLY     │             │
│   │             │          │  Deterministic  │          │             │             │
│   │Deterministic│          │       +         │          │   Skip      │             │
│   │ Navigation  │          │ LLM Navigator   │          │ Navigation  │             │
│   └──────┬──────┘          └────────┬────────┘          └──────┬──────┘             │
│          │                          │                          │                    │
│          └──────────────────────────┼──────────────────────────┘                    │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          RETRIEVE SEEDS                                     │   │
│   │  • Merge structural + semantic seeds                                        │   │
│   │  • Skip semantic search if high-quality structural seeds found              │   │
│   │  • Select top seeds (max 5)                                                 │   │
│   └─────────────────────────────────┬───────────────────────────────────────────┘   │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                       RETRIEVAL LOOP SUB-GRAPH                              │   │
│   │  • Process seeds with adaptive expansion                                    │   │
│   │  • Reflect on progress (query-aware thresholds)                             │   │
│   │  • Synthesize answer with citations                                         │   │
│   │  • Validate and retry if needed                                             │   │
│   └─────────────────────────────────┬───────────────────────────────────────────┘   │
│                                     │                                               │
│                                     ▼                                               │
│   ┌─────────────┐                                                                   │
│   │   Answer    │                                                                   │
│   │ + Citations │                                                                   │
│   └─────────────┘                                                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Flow Diagram

```
                                    START
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         query_agent()            │
                    │  ┌─────────────────────────┐    │
                    │  │ Input:                  │    │
                    │  │ • user_query            │    │
                    │  │ • title_collection      │    │
                    │  │ • text_collection       │    │
                    │  └─────────────────────────┘    │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                                                                    │
│                              ┌──────────────────┐                                  │
│                              │     PLANNER      │                                  │
│                              │                  │                                  │
│                              │  LLM analyzes    │                                  │
│                              │  query to detect │                                  │
│                              │  structural      │                                  │
│                              │  intent          │                                  │
│                              └────────┬─────────┘                                  │
│                                       │                                            │
│                    ┌──────────────────┼──────────────────┐                        │
│                    │                  │                  │                        │
│                    ▼                  ▼                  ▼                        │
│          ┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐               │
│          │   STRUCTURAL    │  │   HYBRID    │  │    SEMANTIC     │               │
│          │     ONLY        │  │             │  │      ONLY       │               │
│          │                 │  │             │  │                 │               │
│          │ chapter=8       │  │ chapter=4   │  │ no structural   │               │
│          │ position=end    │  │ query about │  │ hints           │               │
│          │ keywords=       │  │ "mitosis"   │  │                 │               │
│          │   [review]      │  │             │  │                 │               │
│          └────────┬────────┘  └──────┬──────┘  └────────┬────────┘               │
│                   │                  │                  │                        │
│                   ▼                  ▼                  ▼                        │
│          ┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐               │
│          │  DETERMINISTIC  │  │  LLM-BASED  │  │      SKIP       │               │
│          │   NAVIGATION    │  │  NAVIGATOR  │  │   NAVIGATION    │               │
│          │                 │  │  SUB-GRAPH  │  │                 │               │
│          │ No LLM calls!   │  │             │  │ Go straight to  │               │
│          │ O(1) lookup     │  │ ReAct loop  │  │ semantic search │               │
│          └────────┬────────┘  └──────┬──────┘  └────────┬────────┘               │
│                   │                  │                  │                        │
│                   └──────────────────┼──────────────────┘                        │
│                                      │                                           │
│                                      ▼                                           │
│                         ┌────────────────────────┐                               │
│                         │    RETRIEVE SEEDS      │                               │
│                         │                        │                               │
│                         │ If high-quality        │                               │
│                         │ structural seeds:      │                               │
│                         │ → Skip semantic search │                               │
│                         │                        │                               │
│                         │ Otherwise:             │                               │
│                         │ → Merge both sources   │                               │
│                         └───────────┬────────────┘                               │
│                                     │                                            │
│                                     ▼                                            │
│    ┌────────────────────────────────────────────────────────────────────────┐   │
│    │                     RETRIEVAL LOOP SUB-GRAPH                            │   │
│    │                                                                         │   │
│    │   ┌─────────┐    ┌──────────────┐    ┌───────────┐                     │   │
│    │   │  init   │───▶│ process_seed │───▶│  reflect  │                     │   │
│    │   │  loop   │    │              │    │           │                     │   │
│    │   └─────────┘    │ • grade seed │    │ Query-    │                     │   │
│    │                  │ • adaptive   │    │ aware     │                     │   │
│    │                  │   radius     │    │ thresholds│                     │   │
│    │                  │ • expand &   │    │           │                     │   │
│    │                  │   grade      │    └─────┬─────┘                     │   │
│    │                  │   neighbors  │          │                           │   │
│    │                  └──────────────┘          │                           │   │
│    │                         ▲                  │                           │   │
│    │                         │                  ▼                           │   │
│    │                         │         ┌───────────────┐                    │   │
│    │                  continue?───YES──│ More seeds?   │                    │   │
│    │                                   └───────┬───────┘                    │   │
│    │                                           │ NO                         │   │
│    │                                           ▼                            │   │
│    │                                   ┌───────────────┐                    │   │
│    │                                   │check_evidence │                    │   │
│    │                                   └───────┬───────┘                    │   │
│    │                                           │                            │   │
│    │                            ┌──────────────┴──────────────┐             │   │
│    │                            │                             │             │   │
│    │                            ▼                             ▼             │   │
│    │                    ┌───────────────┐            ┌──────────────┐       │   │
│    │                    │  SUFFICIENT   │            │ INSUFFICIENT │       │   │
│    │                    │               │            │              │       │   │
│    │                    │ structural:   │            │ retry_search │       │   │
│    │                    │  1 high       │            │ (expand      │       │   │
│    │                    │ conceptual:   │            │  scope)      │       │   │
│    │                    │  3 high OR    │            └──────────────┘       │   │
│    │                    │  2h + 3m      │                                   │   │
│    │                    └───────┬───────┘                                   │   │
│    │                            │                                           │   │
│    │                            ▼                                           │   │
│    │                    ┌───────────────┐                                   │   │
│    │                    │  synthesize   │                                   │   │
│    │                    │               │                                   │   │
│    │                    │ Generate      │                                   │   │
│    │                    │ answer with   │                                   │   │
│    │                    │ [Node XXXX]   │                                   │   │
│    │                    │ citations     │                                   │   │
│    │                    └───────┬───────┘                                   │   │
│    │                            │                                           │   │
│    │                            ▼                                           │   │
│    │                    ┌───────────────┐                                   │   │
│    │                    │   validate    │                                   │   │
│    │                    └───────┬───────┘                                   │   │
│    │                            │                                           │   │
│    │              ┌─────────────┼─────────────┐                             │   │
│    │              │             │             │                             │   │
│    │              ▼             ▼             ▼                             │   │
│    │         ┌────────┐   ┌──────────┐   ┌─────────┐                       │   │
│    │         │   OK   │   │ACCEPTABLE│   │  POOR   │                       │   │
│    │         └────┬───┘   └────┬─────┘   └────┬────┘                       │   │
│    │              │            │              │                             │   │
│    │              ▼            ▼              ▼                             │   │
│    │         exit_success  exit_best    retry_synthesis                    │   │
│    │                       _effort      (with feedback)                    │   │
│    │                                                                        │   │
│    └────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │    RESULT     │
                              │               │
                              │ • answer      │
                              │ • evidence    │
                              │ • citations   │
                              └───────────────┘
```

---

## 🧭 Navigation Strategy Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NAVIGATION DECISION TREE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │ Planner detects │
                    │ structural      │
                    │ intent?         │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │   YES    │   │  PARTIAL │   │    NO    │
        │          │   │          │   │          │
        │ chapter  │   │ keywords │   │ pure     │
        │ + keywords│   │ only     │   │ concept  │
        │ + position│   │          │   │          │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │STRUCTURAL│   │  HYBRID  │   │ SEMANTIC │
        │  ONLY    │   │          │   │   ONLY   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             ▼              ▼              ▼
    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │ DETERMINISTIC  │  │ DETERMINISTIC  │  │     SKIP       │
    │ NAVIGATION     │  │ + LLM FALLBACK │  │  NAVIGATION    │
    │                │  │                │  │                │
    │ 1. get_chapter │  │ 1. Try         │  │ Go directly to │
    │    _info()     │  │    deterministic│  │ semantic search│
    │ 2. Filter by   │  │ 2. If fails,   │  │                │
    │    position    │  │    use LLM     │  │                │
    │ 3. Match       │  │    navigator   │  │                │
    │    keywords    │  │                │  │                │
    │                │  │                │  │                │
    │ ⚡ NO LLM CALLS│  │ ~1-8 LLM calls │  │ ⚡ NO LLM CALLS│
    │ ~10ms          │  │ ~5-30 seconds  │  │ ~0ms           │
    └────────────────┘  └────────────────┘  └────────────────┘
```

---

## 📊 Deterministic Navigation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC STRUCTURAL NAVIGATION                           │
│                    (navigate_structural_deterministic)                           │
└─────────────────────────────────────────────────────────────────────────────────┘

 Input: chapter=8, position="end", keywords=["review", "questions"]

                    Step 1: GET CHAPTER INFO
                    ┌─────────────────────────────┐
                    │ get_chapter_info(8)          │
                    │                              │
                    │ Returns:                     │
                    │ {                            │
                    │   chapter: 8,                │
                    │   start_idx: 331,            │
                    │   end_idx: 368,              │
                    │   num_nodes: 37,             │
                    │   sections: [...]            │
                    │ }                            │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    Step 2: FILTER BY POSITION
                    ┌─────────────────────────────┐
                    │ position = "end"             │
                    │                              │
                    │ Full chapter: nodes 331-368  │
                    │ ────────────────────────────│
                    │ [331] Chapter 8 header       │
                    │ [332] Section 8.1            │
                    │ [333] ...                    │
                    │ ...                          │
                    │ [358] ─────────────────────  │
                    │ [359] │ Last 10 nodes │ ◄── position="end"
                    │ [360] │ searched here │      │
                    │ [361] │               │      │
                    │ [362] │ Review        │      │
                    │ [363] │ Questions     │      │
                    │ [364] │               │      │
                    │ [365] │               │      │
                    │ [366] │               │      │
                    │ [367] │               │      │
                    │ [368] └───────────────┘      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    Step 3: MATCH KEYWORDS
                    ┌─────────────────────────────┐
                    │ keywords = ["review",        │
                    │             "questions"]     │
                    │                              │
                    │ Scan titles & text for:      │
                    │ • "review" in title? ✓ HIGH  │
                    │ • "questions" in title? ✓    │
                    │ • keywords in text? MEDIUM   │
                    │                              │
                    │ Found 3 matches!             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    Step 4: RETURN RESULTS
                    ┌─────────────────────────────┐
                    │ Returns:                     │
                    │ [                            │
                    │   {node_id: "0366",          │
                    │    title: "Review Questions",│
                    │    relevance_grade: "high",  │
                    │    source: "structural"},    │
                    │   {node_id: "0367", ...},    │
                    │   {node_id: "0368", ...}     │
                    │ ]                            │
                    │                              │
                    │ ⚡ Total time: ~10ms         │
                    │ ⚡ LLM calls: 0              │
                    └─────────────────────────────┘
```

---

## 🔁 Navigator Sub-graph (LLM-based)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       NAVIGATOR SUB-GRAPH (for hybrid queries)                   │
│                             ReAct-style tool calling                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │ parse_goal  │
                              │             │
                              │ Build NL    │
                              │ goal from   │
                              │ hints       │
                              └──────┬──────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────┐
              │              execute_tools              │
              │                                         │
              │  Available tools:                       │
              │  • nav_search_title(query, top_k)       │
              │  • nav_explore_titles(node_id, dir)     │
              │  • nav_peek_content(node_id)            │
              │  • nav_mark_found(node_ids)             │
              │                                         │
              │  LLM decides which tool to call         │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │                reflect                   │
              │                                         │
              │  Evaluate progress:                     │
              │  • Making progress? (yes/no/uncertain)  │
              │  • Next action? (continue/verify/       │
              │                  expand_scope/give_up)  │
              └───────────────────┬─────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
    │   searching   │    │   verifying   │    │    retry      │
    │               │    │               │    │               │
    │ Continue      │    │ Check if      │    │ Expand scope  │
    │ exploring     │    │ candidates    │    │ and restart   │
    └───────┬───────┘    │ match goal    │    └───────────────┘
            │            └───────┬───────┘
            │                    │
            ▼                    ▼
    execute_tools ◄──────   ┌───────────────┐
                            │    verify     │
                            │               │
                            │ Check keyword │
                            │ matches in    │
                            │ found nodes   │
                            └───────┬───────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                 ┌───────────────┐    ┌───────────────┐
                 │    found      │    │   not found   │
                 │               │    │               │
                 │ exit_success  │    │ Continue      │
                 │               │    │ searching     │
                 └───────────────┘    └───────────────┘
```

---

## 📈 Evidence Thresholds by Query Type

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      QUERY-AWARE EVIDENCE THRESHOLDS                             │
│                      (Anthropic: "Use explicit criteria")                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬────────────────────────────────┬────────────────────────────┐
│    Query Type    │      is_evidence_sufficient    │       should_early_exit    │
├──────────────────┼────────────────────────────────┼────────────────────────────┤
│                  │                                │                            │
│   STRUCTURAL     │  high ≥ 1                      │  high ≥ 2                  │
│   (route:        │       OR                       │       OR                   │
│  structural_only)│  medium ≥ 2                    │  high ≥ 1 AND medium ≥ 2   │
│                  │                                │                            │
│  "Chapter 8      │  ✓ Found the exact section    │  ✓ Found section + context │
│   review         │    we were looking for         │                            │
│   questions"     │                                │                            │
│                  │                                │                            │
├──────────────────┼────────────────────────────────┼────────────────────────────┤
│                  │                                │                            │
│     HYBRID       │  high ≥ 2                      │  high ≥ 3                  │
│   (route:        │       OR                       │       OR                   │
│    hybrid)       │  high ≥ 1 AND medium ≥ 3       │  high ≥ 2 AND medium ≥ 3   │
│                  │                                │                            │
│  "Explain        │  ✓ Found section + enough      │  ✓ Good coverage           │
│   mitosis from   │    conceptual context          │                            │
│   chapter 4"     │                                │                            │
│                  │                                │                            │
├──────────────────┼────────────────────────────────┼────────────────────────────┤
│                  │                                │                            │
│   CONCEPTUAL     │  high ≥ 3                      │  high ≥ 5                  │
│   (route:        │       OR                       │       OR                   │
│  semantic_only)  │  high ≥ 2 AND medium ≥ 3       │  high ≥ 3 AND medium ≥ 4   │
│                  │       OR                       │                            │
│                  │  medium ≥ 8                    │                            │
│                  │                                │                            │
│  "What is        │  ✓ Need broad coverage of      │  ✓ Excellent evidence      │
│   photosynthesis?│    the concept                 │    collected               │
│   "              │                                │                            │
│                  │                                │                            │
└──────────────────┴────────────────────────────────┴────────────────────────────┘
```

---

## 🔄 RetrievalLoop Sub-graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LOOP SUB-GRAPH                                 │
│                         ReAct-style seed processing                              │
└─────────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │   init_loop    │
                        │                │
                        │ Initialize:    │
                        │ • pending_seeds│
                        │ • evidence_pool│
                        │ • scope_level=0│
                        └───────┬────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │        process_seed          │
                 │                              │
                 │ 1. Pop seed from pending     │
                 │ 2. Grade seed (LLM call)     │
                 │                              │
                 │ If low-grade:                │
                 │   → Skip to next seed        │
                 │                              │
                 │ If high/medium:              │
                 │   3. Calculate radius        │
                 │      high: base + 2          │
                 │      medium: base            │
                 │   4. Expand around seed      │
                 │   5. Grade neighbors (batch) │
                 │   6. Add to evidence_pool    │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │           reflect            │
                 │                              │
                 │ Check query-aware thresholds:│
                 │                              │
                 │ structural_only?             │
                 │   → 1 high is enough         │
                 │                              │
                 │ conceptual?                  │
                 │   → Need 3 high OR 2h+3m     │
                 │                              │
                 │ Early exit if excellent      │
                 │ evidence found               │
                 └──────────────┬───────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        ┌───────────┐    ┌───────────┐    ┌───────────┐
        │ continue  │    │ sufficient│    │insufficient│
        │           │    │           │    │           │
        │ More      │    │ check_    │    │ retry_    │
        │ seeds     │    │ evidence  │    │ search    │
        │ remain    │    │           │    │           │
        └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
              │                │                 │
              │                │                 │
              ▼                ▼                 ▼
        process_seed    synthesize      ┌───────────────┐
                                        │ Expand scope  │
                                        │               │
                                        │ scope 0→1→2   │
                                        │ focused →     │
                                        │ expanded →    │
                                        │ broad         │
                                        │               │
                                        │ Get new seeds │
                                        │ with higher   │
                                        │ top_k         │
                                        └───────┬───────┘
                                                │
                                                ▼
                                          process_seed


                 ┌──────────────────────────────┐
                 │          synthesize          │
                 │                              │
                 │ Build evidence context:      │
                 │ • Full text for high-grade   │
                 │ • 5000 chars for medium      │
                 │                              │
                 │ Generate answer with         │
                 │ [Node XXXX] citations        │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │           validate           │
                 │                              │
                 │ Check:                       │
                 │ • Every sentence has citation│
                 │ • All node IDs are valid     │
                 │ • Answer is non-empty        │
                 └──────────────┬───────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        ┌───────────┐    ┌───────────┐    ┌───────────┐
        │    OK     │    │ ACCEPTABLE│    │   POOR    │
        │           │    │           │    │           │
        │ exit_     │    │ exit_best │    │ retry_    │
        │ success   │    │ _effort   │    │ synthesis │
        │           │    │           │    │           │
        │ Return    │    │ Return    │    │ Add       │
        │ validated │    │ best      │    │ feedback  │
        │ answer    │    │ answer    │    │ → retry   │
        └───────────┘    └───────────┘    └─────┬─────┘
                                                │
                                                ▼
                                          synthesize
                                          (with feedback)
```

---

## 📋 State Definitions

### AgentState (Main Graph)

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    title_collection: str
    text_collection: str
    final_answer: str
    plan: Dict[str, Any]          # Contains: route, query_type, structural_hints
    validation: Dict[str, Any]
    structural_seeds: List[Dict]   # From navigator
    pending_seeds: List[Dict]      # For retrieval loop
    visited_node_ids: List[str]
    evidence_pool: List[Dict]
    processing_complete: bool
```

### Plan Structure

```python
plan = {
    "route": "structural_only" | "semantic_only" | "hybrid",
    "query_type": "structural" | "conceptual" | "hybrid",
    "has_structural": bool,
    "structural_hints": {
        "chapter": int | None,
        "position": "beginning" | "end" | None,
        "section_keywords": ["review", "questions", ...],
        "has_structural": bool,
    },
    "search_query": "stripped query for semantic search",
    "subqueries": ["search_query"],
    "top_k": 8,
    "radius": 3,
    "max_seeds": 5,
}
```

---

## ⚡ Performance Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PERFORMANCE COMPARISON                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┬─────────────────────────┬─────────────────────────────────┐
│       Metric         │    Before (LLM-based)   │   After (Anthropic-style)       │
├──────────────────────┼─────────────────────────┼─────────────────────────────────┤
│                      │                         │                                 │
│ Structural Query     │ • Navigator: 8-24 calls │ • Deterministic: 0 LLM calls    │
│ Navigation Time      │ • Time: 15-45 seconds   │ • Time: ~10ms                   │
│                      │ • Unpredictable         │ • Predictable                   │
│                      │                         │                                 │
├──────────────────────┼─────────────────────────┼─────────────────────────────────┤
│                      │                         │                                 │
│ Conceptual Query     │ • Always runs navigator │ • Skips navigation entirely     │
│ Navigation Time      │ • Time: 3-10 seconds    │ • Time: ~0ms                    │
│                      │   (even if no results)  │                                 │
│                      │                         │                                 │
├──────────────────────┼─────────────────────────┼─────────────────────────────────┤
│                      │                         │                                 │
│ Evidence Check       │ • Same thresholds for   │ • Query-aware thresholds        │
│                      │   all query types       │ • Structural: 1 high enough     │
│                      │ • Often over-retrieves  │ • Conceptual: needs coverage    │
│                      │                         │                                 │
├──────────────────────┼─────────────────────────┼─────────────────────────────────┤
│                      │                         │                                 │
│ API Calls            │ • Planner: 1            │ • Planner: 1                    │
│ (Structural Query)   │ • Navigator: 8-24       │ • Navigator: 0 (deterministic)  │
│                      │ • Retrieval: 10-30      │ • Retrieval: 5-15               │
│                      │ • Total: 20-55          │ • Total: 6-16                   │
│                      │                         │                                 │
├──────────────────────┼─────────────────────────┼─────────────────────────────────┤
│                      │                         │                                 │
│ Total Time           │ • Structural: 30-90s    │ • Structural: 10-25s            │
│                      │ • Conceptual: 20-60s    │ • Conceptual: 15-40s            │
│                      │                         │                                 │
└──────────────────────┴─────────────────────────┴─────────────────────────────────┘
```

---

## 🏗️ Graph Structure Summary

```
Main StateGraph
├── Entry Point: "planner"
├── Nodes:
│   ├── "planner" → planner() 
│   │   └── Detects intent, sets route
│   ├── "navigate" → navigate_structure()
│   │   ├── route=structural_only → deterministic navigation
│   │   ├── route=hybrid → LLM navigator sub-graph  
│   │   └── route=semantic_only → skip (return [])
│   ├── "retrieve" → retrieve_seeds()
│   │   └── Merges structural + semantic seeds
│   └── "retrieval_loop" → run_retrieval_loop()
│       └── Invokes RetrievalLoop sub-graph
└── Edges:
    ├── planner → navigate
    ├── navigate → retrieve  
    ├── retrieve → retrieval_loop
    └── retrieval_loop → END

Navigator Sub-graph (for hybrid queries)
├── Entry Point: "parse_goal"
├── Nodes: parse_goal, execute_tools, reflect, verify, 
│          retry_broader, exit_success, exit_failure
└── Tools: nav_search_title, nav_explore_titles, 
           nav_peek_content, nav_mark_found

RetrievalLoop Sub-graph
├── Entry Point: "init_loop"
├── Nodes: init_loop, process_seed, reflect, check_evidence,
│          retry_search, synthesize, validate, retry_synthesis,
│          exit_success, exit_best_effort
└── Features: Query-aware thresholds, adaptive radius,
              scope expansion, synthesis retry
```

---

## 🎯 Anthropic Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **"Start Simple"** | Deterministic navigation for clear structural queries |
| **"Routing"** | Planner routes to appropriate handler based on query type |
| **"Clear ACI"** | `get_chapter_info()` exposes document structure clearly |
| **"Explicit Criteria"** | Query-aware evidence thresholds, not always asking LLM |
| **"Control Costs"** | Skip unnecessary operations when possible |
| **"Prompt Chaining"** | Sequential steps in deterministic navigation |

---

## 📝 Example Executions

### Example 1: Structural Query

```
Query: "What are the review questions at the end of chapter 8?"

┌─ PLANNER ─────────────────────────────────────────────────────────┐
│ query_type: structural                                            │
│ route: structural_only                                            │
│ structural_hints:                                                 │
│   chapter: 8                                                      │
│   position: end                                                   │
│   keywords: [review, questions]                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ NAVIGATE (DETERMINISTIC) ────────────────────────────────────────┐
│ get_chapter_info(8) → nodes 331-368                               │
│ Filter by position: last 10 nodes                                 │
│ Match keywords: 3 matches found                                   │
│ Time: ~10ms | LLM calls: 0                                        │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RETRIEVE ────────────────────────────────────────────────────────┐
│ High-quality structural seeds found                               │
│ → Skip semantic search                                            │
│ Seeds: 3 structural, 0 semantic                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RETRIEVAL LOOP ──────────────────────────────────────────────────┐
│ Process seed 0366: grade=low → skip                               │
│ Process seed 0367: grade=medium → expand                          │
│ Evidence: high=1, medium=1                                        │
│ Threshold (structural): 1 high OR 2 medium ✓                      │
│ → Sufficient! Proceed to synthesis                                │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RESULT ──────────────────────────────────────────────────────────┐
│ Answer with review questions from Chapter 8                       │
│ Citations: [Node 0366], [Node 0367]                               │
│ Total time: ~25 seconds                                           │
│ Total LLM calls: ~8 (planner + grading + synthesis)               │
└───────────────────────────────────────────────────────────────────┘
```

### Example 2: Conceptual Query

```
Query: "What is the difference between dominant and recessive traits?"

┌─ PLANNER ─────────────────────────────────────────────────────────┐
│ query_type: conceptual                                            │
│ route: semantic_only                                              │
│ structural_hints: None                                            │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ NAVIGATE ────────────────────────────────────────────────────────┐
│ Route = semantic_only                                             │
│ → SKIP NAVIGATION                                                 │
│ Time: ~0ms | LLM calls: 0                                         │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RETRIEVE ────────────────────────────────────────────────────────┐
│ Semantic search with query                                        │
│ Seeds: 0 structural, 5 semantic                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RETRIEVAL LOOP ──────────────────────────────────────────────────┐
│ Process seed 0342: grade=high → expand with radius+2              │
│ Evidence: high=2, medium=3                                        │
│ Threshold (conceptual): 3 high OR 2h+3m ✓                         │
│ → Sufficient! Proceed to synthesis                                │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ RESULT ──────────────────────────────────────────────────────────┐
│ Answer explaining dominant vs recessive traits                    │
│ Citations: [Node 0342], [Node 0340], [Node 0343], ...             │
│ Total time: ~20 seconds                                           │
│ Total LLM calls: ~10 (planner + grading + synthesis)              │
└───────────────────────────────────────────────────────────────────┘
```

---

*Last updated: January 2026*
*Architecture version: 2.0 (Anthropic-style routing)*
