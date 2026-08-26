#!/usr/bin/env python3
"""
Fast Non-LLM 100-Question Retrieval & Context Purification Benchmark Harness.
Evaluates Layer 1 (Retrieval, Hit Rate, MRR, nDCG, MAP, Routing) and Layer 2
(Information Density, Overlap, Boundary Completeness, Table Integrity, In-Context Keywords)
across the unified 100 benchmark questions with ZERO LLM API tokens and ZERO API key dependency.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.retrieval.hybrid_search import HybridSearchEngine
from rag_project.retrieval.reranker import CrossEncoderReranker
from rag_project.retrieval.neighbor_expander import NeighborExpander
from rag_project.retrieval.overlap_merger import OverlapMerger
from rag_project.retrieval.adaptive_strategy import AdaptiveRetrievalStrategy
from rag_project.retrieval.doc_router import MultiDocRouter, normalize_digits

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def load_unified_benchmark(benchmark_file: Path) -> List[Dict[str, Any]]:
    """Load unified 100 benchmark test cases."""
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found at: {benchmark_file}")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])


def extract_pages_from_citation(citation_str: str) -> List[int]:
    """Extract integer page numbers from ground-truth citation strings."""
    pages = []
    for m in re.finditer(r"(?:pages?|p\.)\s*(\d+(?:\s*[-–]\s*\d+)?)", citation_str, re.IGNORECASE):
        span = m.group(1).replace("–", "-")
        if "-" in span:
            parts = span.split("-")
            try:
                start_p, end_p = int(parts[0].strip()), int(parts[1].strip())
                pages.extend(range(start_p, end_p + 1))
            except Exception:
                pass
        else:
            try:
                pages.append(int(span.strip()))
            except Exception:
                pass
    return list(set(pages))


def normalize_hardware_term(term: str) -> str:
    """Normalize hardware attributes, pin names, and bus widths."""
    t = term.lower().strip()
    t = re.sub(r"[\[\]\(\)\-_:,\.]", " ", t)
    return " ".join(t.split())


def evaluate_retrieval_ranking_metrics(
    sources: List[Dict[str, Any]],
    target_docs: List[str],
    expected_pages: List[int],
    k_list: List[int] = [5, 10, 15, 25]
) -> Dict[str, Any]:
    """Calculate Hit@K, MRR@K, nDCG@K, MAP@K for retrieval ranking."""
    doc_aliases = {"pg063": "pg036", "pg036": "pg036", "pg058": "pg058", "pg065": "pg065", "ug380": "ug380", "ug389": "ug389", "ug682": "ug682"}
    norm_targets = set(doc_aliases.get(d.lower(), d.lower()) for d in target_docs)

    relevance_scores = []
    first_golden_rank = None

    for rank_idx, s in enumerate(sources, start=1):
        s_doc = doc_aliases.get(str(s.get("doc_id", "")).lower(), str(s.get("doc_id", "")).lower())
        s_page = s.get("page_number")
        
        is_target_doc = s_doc in norm_targets
        is_target_page = (s_page in expected_pages) if expected_pages else is_target_doc

        if is_target_doc and is_target_page:
            relevance_scores.append(3)
            if first_golden_rank is None:
                first_golden_rank = rank_idx
        elif is_target_doc:
            relevance_scores.append(1)
            if first_golden_rank is None:
                first_golden_rank = rank_idx
        else:
            relevance_scores.append(0)

    metrics = {}
    for k in k_list:
        sub_rels = relevance_scores[:k]
        
        # Hit Rate @ K
        metrics[f"hit_rate_at_{k}"] = 1.0 if any(r > 0 for r in sub_rels) else 0.0
        
        # MRR @ K
        if first_golden_rank is not None and first_golden_rank <= k:
            metrics[f"mrr_at_{k}"] = round(1.0 / first_golden_rank, 4)
        else:
            metrics[f"mrr_at_{k}"] = 0.0
        
        # nDCG @ K
        dcg = sum((2**r - 1) / math.log2(idx + 1) for idx, r in enumerate(sub_rels, start=1))
        ideal_rels = sorted(relevance_scores, reverse=True)[:k]
        if not any(ideal_rels):
            ideal_rels = [3]
        idcg = sum((2**r - 1) / math.log2(idx + 1) for idx, r in enumerate(ideal_rels, start=1))
        metrics[f"ndcg_at_{k}"] = round(dcg / idcg, 4) if idcg > 0 else 0.0

        # MAP @ K
        cum_precisions = []
        hits_so_far = 0
        for idx, r in enumerate(sub_rels, start=1):
            if r > 0:
                hits_so_far += 1
                cum_precisions.append(hits_so_far / idx)
        metrics[f"map_at_{k}"] = round(sum(cum_precisions) / max(1, min(len(norm_targets), k)), 4) if cum_precisions else 0.0

    return metrics


def evaluate_context_purification_metrics(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure Information Density, Exact Overlap Ratio, Boundary Completeness, Table Integrity."""
    if not sources:
        return {
            "information_density_pct": 100.0,
            "exact_overlap_ratio_pct": 0.0,
            "boundary_completeness_pct": 100.0,
            "table_integrity_pct": 100.0,
            "total_context_tokens": 0,
            "chunk_count": 0
        }

    total_tokens = 0
    all_words = []
    complete_boundaries = 0
    tables_found = 0
    valid_tables = 0
    duplicate_ngrams = 0
    seen_ngrams = set()

    for s in sources:
        content = s.get("content", "")
        tokens = content.split()
        total_tokens += len(tokens)
        all_words.extend(tokens)

        # Boundary Completeness
        stripped = content.strip()
        if bool(re.search(r"[\.!\?:\`\|\"]\s*$", stripped)):
            complete_boundaries += 1

        # Table Structural Integrity
        if "|" in content and "\n" in content:
            tables_found += 1
            table_lines = [l.strip() for l in content.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
            if len(table_lines) >= 2:
                col_counts = [len(l.split("|")) for l in table_lines]
                if max(col_counts) - min(col_counts) <= 1:
                    valid_tables += 1

        # Exact Overlap (5-gram repetition check across distinct chunks)
        for i in range(len(tokens) - 4):
            ngram = " ".join(tokens[i:i+5]).lower()
            if ngram in seen_ngrams:
                duplicate_ngrams += 5
            else:
                seen_ngrams.add(ngram)

    unique_tokens = len(set(w.lower() for w in all_words))
    novelty_pct = round((unique_tokens / max(1, total_tokens)) * 100, 2)
    overlap_pct = round((duplicate_ngrams / max(1, total_tokens)) * 100, 2)
    boundary_pct = round((complete_boundaries / max(1, len(sources))) * 100, 2)
    table_pct = round((valid_tables / max(1, tables_found)) * 100, 2) if tables_found > 0 else 100.0

    return {
        "information_density_pct": novelty_pct,
        "exact_overlap_ratio_pct": min(100.0, overlap_pct),
        "boundary_completeness_pct": boundary_pct,
        "table_integrity_pct": table_pct,
        "total_context_tokens": total_tokens,
        "chunk_count": len(sources)
    }


def evaluate_keyword_coverage_in_context(eval_keywords: List[str], context_text: str) -> Tuple[float, List[str], List[str]]:
    """Measure what percentage of critical technical keywords/pins exist in the retrieved context."""
    if not eval_keywords:
        return 1.0, [], []

    ctx_norm = normalize_hardware_term(context_text.lower())
    matched = []
    missing = []

    for kw in eval_keywords:
        kw_norm = normalize_hardware_term(kw.lower())
        if kw_norm in ctx_norm:
            matched.append(kw)
        else:
            tokens = [t for t in kw_norm.split() if len(t) > 1]
            if tokens and all(t in ctx_norm for t in tokens):
                matched.append(kw)
            else:
                missing.append(kw)

    coverage = round(len(matched) / len(eval_keywords), 4)
    return coverage, matched, missing


def run_benchmark():
    benchmark_file = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_100_unified.json"
    if not benchmark_file.exists():
        benchmark_file = PROJECT_ROOT.parent / "evaluation" / "datasets" / "benchmark_100_unified.json"

    test_cases = load_unified_benchmark(benchmark_file)
    print(f"Loaded {len(test_cases)} test cases from {benchmark_file.name}.", flush=True)

    hybrid_engine = HybridSearchEngine()
    reranker = CrossEncoderReranker()
    neighbor_expander = NeighborExpander()
    overlap_merger = OverlapMerger()
    adaptive_strategy = AdaptiveRetrievalStrategy()

    doc_aliases = {"pg063": "pg036", "pg036": "pg036", "pg058": "pg058", "pg065": "pg065", "ug380": "ug380", "ug389": "ug389", "ug682": "ug682"}

    results = []
    t_start = time.time()

    for idx, tc in enumerate(test_cases, 1):
        q = tc["question"]
        target_docs = tc.get("target_docs", [])
        norm_targets = [doc_aliases.get(d.lower(), d.lower()) for d in target_docs]
        eval_keywords = tc.get("eval_criteria_keywords", [])
        citations_expected = tc.get("citations", [])

        expected_pages = []
        for c_str in citations_expected:
            expected_pages.extend(extract_pages_from_citation(c_str))

        t0 = time.time()
        route_info = MultiDocRouter.route_query(q)
        effective_doc_filter = route_info.get("target_docs")

        # Dynamic retrieval parameters
        req_k = 25
        adaptive_cfg = adaptive_strategy.analyze_and_adapt(q, base_k=req_k)
        effective_k = adaptive_cfg.get("effective_k", req_k)
        port_boost = adaptive_cfg.get("port_boost", 0.0)

        # Stage 1: Hybrid Retrieval
        candidates = hybrid_engine.search(
            q,
            top_candidates=max(60, effective_k * 2),
            top_k_dense=max(60, effective_k * 2),
            top_k_sparse=max(60, effective_k * 2),
            doc_filter=effective_doc_filter
        )

        # Stage 2: Cross-Encoder Reranking
        reranked = reranker.rerank(
            query=q,
            candidates=candidates,
            top_k=effective_k,
            hardware_port_boost=port_boost,
            target_docs=effective_doc_filter
        )

        # Stage 3: Sequential Neighbor Expansion
        expanded = neighbor_expander.expand_neighbors(
            ranked_chunks=reranked,
            target_docs=effective_doc_filter
        )

        # Stage 4: Overlap Reducer
        dedup_res = overlap_merger.reduce_chunks(expanded)
        final_chunks = dedup_res.get("reduced_chunks", expanded)
        retrieval_ms = round((time.time() - t0) * 1000, 2)

        context_body = "\n\n".join([c.get("content", "") for c in final_chunks])

        # Layer 1 Metrics
        retrieval_metrics = evaluate_retrieval_ranking_metrics(final_chunks, target_docs, expected_pages)

        # Routing & Page Recall
        retrieved_docs = set(doc_aliases.get(str(s.get("doc_id", "")).lower(), str(s.get("doc_id", "")).lower()) for s in final_chunks)
        doc_hits = sum(1 for d in norm_targets if d in retrieved_docs)
        doc_routing_acc = (doc_hits / len(norm_targets)) if norm_targets else 1.0

        retrieved_pages = [s.get("page_number") for s in final_chunks if s.get("page_number") is not None]
        page_hit = any(ep in retrieved_pages for ep in expected_pages) if expected_pages else True

        # Layer 2 Metrics
        purif_metrics = evaluate_context_purification_metrics(final_chunks)

        # Layer 3 In-Context Keyword Coverage
        kw_cov, matched_kw, missing_kw = evaluate_keyword_coverage_in_context(eval_keywords, context_body)

        results.append({
            "id": tc["id"],
            "step": tc["step"],
            "doc_routing_accuracy": doc_routing_acc,
            "page_hit": page_hit,
            "retrieval_latency_ms": retrieval_ms,
            "keyword_in_context_coverage": kw_cov,
            **retrieval_metrics,
            **purif_metrics
        })

        if idx % 10 == 0 or idx == len(test_cases):
            print(f"[{idx}/{len(test_cases)}] Evaluated: {tc['id']} | Routing: {doc_routing_acc*100:.0f}% | Latency: {retrieval_ms:.1f}ms", flush=True)

    total_time = round(time.time() - t_start, 2)
    macro = {
        "total_cases": len(results),
        "total_evaluation_time_sec": total_time,
        "avg_retrieval_latency_ms": round(sum(r["retrieval_latency_ms"] for r in results) / len(results), 2),
        "macro_hit_rate_at_5": round(sum(r["hit_rate_at_5"] for r in results) / len(results) * 100, 2),
        "macro_hit_rate_at_10": round(sum(r["hit_rate_at_10"] for r in results) / len(results) * 100, 2),
        "macro_hit_rate_at_25": round(sum(r["hit_rate_at_25"] for r in results) / len(results) * 100, 2),
        "macro_mrr_at_5": round(sum(r["mrr_at_5"] for r in results) / len(results), 4),
        "macro_mrr_at_25": round(sum(r["mrr_at_25"] for r in results) / len(results), 4),
        "macro_ndcg_at_10": round(sum(r["ndcg_at_10"] for r in results) / len(results), 4),
        "macro_ndcg_at_25": round(sum(r["ndcg_at_25"] for r in results) / len(results), 4),
        "macro_map_at_10": round(sum(r["map_at_10"] for r in results) / len(results), 4),
        "macro_map_at_25": round(sum(r["map_at_25"] for r in results) / len(results), 4),
        "macro_doc_routing_accuracy_pct": round(sum(r["doc_routing_accuracy"] for r in results) / len(results) * 100, 2),
        "macro_page_hit_rate_pct": round(sum(1.0 if r["page_hit"] else 0.0 for r in results) / len(results) * 100, 2),
        "macro_information_density_pct": round(sum(r["information_density_pct"] for r in results) / len(results), 2),
        "macro_exact_overlap_ratio_pct": round(sum(r["exact_overlap_ratio_pct"] for r in results) / len(results), 2),
        "macro_boundary_completeness_pct": round(sum(r["boundary_completeness_pct"] for r in results) / len(results), 2),
        "macro_table_integrity_pct": round(sum(r["table_integrity_pct"] for r in results) / len(results), 2),
        "macro_avg_context_tokens": round(sum(r["total_context_tokens"] for r in results) / len(results), 1),
        "macro_keyword_in_context_coverage_pct": round(sum(r["keyword_in_context_coverage"] for r in results) / len(results) * 100, 2),
    }

    print("\n" + "="*80, flush=True)
    print("FAST RETRIEVAL & CONTEXT PURIFICATION BENCHMARK REPORT (100 QUESTIONS)", flush=True)
    print("="*80, flush=True)
    print(json.dumps(macro, indent=2), flush=True)


if __name__ == "__main__":
    run_benchmark()
