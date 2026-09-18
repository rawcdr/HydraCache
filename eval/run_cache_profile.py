import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

from src.retrieval.pipeline import PipelineRetriever
from src.retrieval.answer import AnswerPipeline
from src.cache.semantic_cache import SemanticCache

def main():
    load_dotenv()
    
    with open("eval/test_set.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    pipeline = PipelineRetriever()
    cache = SemanticCache()
    cache.clear()
    
    app = AnswerPipeline(pipeline=pipeline, cache=cache)
    
    # Workload Simulation
    # 1. First 10 questions (should be MISS)
    # 2. Same 10 questions exactly (should be HIT)
    # 3. Same 10 questions slightly rephrased (should be HIT)
    # 4. Next 10 questions (should be MISS)
    
    workload = []
    
    # Phase 1: 10 Misses
    for i in range(10):
        workload.append({"q": qa_pairs[i]["question"], "type": "Novel"})
        
    # Phase 2: 10 Exact Hits
    for i in range(10):
        workload.append({"q": qa_pairs[i]["question"], "type": "Exact Duplicate"})
        
    # Phase 3: 10 Semantic Hits
    rephrasings = [
        "What was the total net sales for Americas in 2025?",
        "How much net sales did Europe have in 2025?",
        "Tell me Greater China's total net sales for 2024.",
        "What date did DOJ file the antitrust lawsuit?",
        "For 2025, what was the RSU vesting-date fair value?",
        "List the supply chain risks mentioned in the 2025 10-K.",
        "What channels does Apple use to distribute products?",
        "Summarize the 2024 DOJ antitrust lawsuit.",
        "What are the responsibilities of the Head of Corporate Information Security?",
        "What are the consequences if Apple's culture and workforce dynamics fail?"
    ]
    for i, q in enumerate(rephrasings):
        workload.append({"q": q, "type": "Semantic Duplicate"})
        
    # Phase 4: 10 Novel
    for i in range(10, 20):
        workload.append({"q": qa_pairs[i]["question"], "type": "Novel"})
        
    print(f"Executing Workload Profile (Total: {len(workload)} queries)...")
    
    results = []
    
    for i, item in enumerate(workload):
        print(f"[{i+1}/{len(workload)}] {item['type']} -> {item['q']}")
        res = app.answer(item['q'])
        
        results.append({
            "query": item['q'],
            "type": item['type'],
            "cache_hit": res.cache_hit,
            "cache_distance": res.cache_distance,
            "lookup_latency": res.cache_latency,
            "retrieval_latency": res.retrieval_latency,
            "generation_latency": res.generation_latency,
            "total_latency": res.total_latency,
            "total_tokens": res.total_tokens
        })
        
    df = pd.DataFrame(results)
    
    print("\n==============================")
    print("CACHE PERFORMANCE SUMMARY")
    print("==============================\n")
    
    hit_rate = df['cache_hit'].mean() * 100
    print(f"Overall Cache Hit Rate: {hit_rate:.1f}%")
    
    exact_hit_rate = df[df['type'] == 'Exact Duplicate']['cache_hit'].mean() * 100
    print(f"Exact Duplicate Hit Rate: {exact_hit_rate:.1f}%")
    
    semantic_hit_rate = df[df['type'] == 'Semantic Duplicate']['cache_hit'].mean() * 100
    print(f"Semantic Duplicate Hit Rate: {semantic_hit_rate:.1f}%")
    
    novel_miss_rate = (1.0 - df[df['type'] == 'Novel']['cache_hit'].mean()) * 100
    print(f"Novel Query Miss Rate: {novel_miss_rate:.1f}%")
    
    print("\n--- Latency Profiles (ms) ---")
    
    hits = df[df['cache_hit'] == True]
    misses = df[df['cache_hit'] == False]
    
    if len(hits) > 0:
        print(f"Cache HIT  p50 Latency: {hits['total_latency'].quantile(0.5):.2f} ms")
        print(f"Cache HIT  p95 Latency: {hits['total_latency'].quantile(0.95):.2f} ms")
    else:
        print("No Cache Hits.")
        
    if len(misses) > 0:
        print(f"Cache MISS p50 Latency: {misses['total_latency'].quantile(0.5):.2f} ms")
        print(f"Cache MISS p95 Latency: {misses['total_latency'].quantile(0.95):.2f} ms")
        print(f"Cache MISS p50 Generation: {misses['generation_latency'].quantile(0.5):.2f} ms")
        print(f"Cache MISS p50 Retrieval: {misses['retrieval_latency'].quantile(0.5):.2f} ms")
    else:
        print("No Cache Misses.")
        
    df.to_csv("eval/cache_profile.csv", index=False)
    print("\nRaw data saved to eval/cache_profile.csv")

if __name__ == "__main__":
    main()
