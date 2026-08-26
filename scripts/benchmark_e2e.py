#!/usr/bin/env python3
"""
End-to-End LLM Generation Benchmark Runner across Unified 100 Benchmark Questions.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.generation.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_e2e_benchmark(max_questions: int = 100, provider: str = "gemini", model: str = "gemini-3.5-flash-lite"):
    benchmark_file = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_100_unified.json"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    test_cases = data.get("test_cases", [])[:max_questions]
    logger.info(f"Running E2E benchmark on {len(test_cases)} questions (Provider: {provider}, Model: {model})...")

    pipeline = RAGPipeline(provider=provider, model=model)
    results = []

    for idx, tc in enumerate(test_cases, 1):
        q = tc["question"]
        logger.info(f"[{idx}/{len(test_cases)}] Executing: {q[:60]}...")
        t0 = time.time()
        res = pipeline.query(q, auto_k=True)
        latency = round((time.time() - t0) * 1000, 2)

        results.append({
            "id": tc["id"],
            "step": tc["step"],
            "question": q,
            "answer": res.get("answer"),
            "sources_count": len(res.get("sources", [])),
            "timings_ms": res.get("timings_ms", {}),
            "total_latency_ms": latency
        })

    out_file = PROJECT_ROOT / "evaluation" / "datasets" / "iterations" / "e2e_benchmark_run.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"E2E Benchmark completed! Results saved to {out_file}")


if __name__ == "__main__":
    run_e2e_benchmark()
