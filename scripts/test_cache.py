import time
import sys
from src.retrieval.answer import AnswerPipeline

def print_result(query_label: str, query: str, res):
    print(f"\n[{query_label}]")
    print(f"Query: '{query}'")
    print(f"Cache Hit: {res.cache_hit}")
    if res.cache_distance is not None:
        print(f"Cache Distance: {res.cache_distance:.4f}")
    else:
        print(f"Cache Distance: N/A")
    print(f"Cache Lookup Latency: {res.cache_latency:.2f} ms")
    print(f"Retrieval Latency: {res.retrieval_latency:.2f} ms")
    print(f"Generation Latency: {res.generation_latency:.2f} ms")
    print(f"Total Latency: {res.total_latency:.2f} ms")
    if res.total_tokens > 0:
        print(f"Token Usage: Prompt={res.prompt_tokens}, Completion={res.completion_tokens}, Total={res.total_tokens}")
    print("-" * 50)
    print(f"Answer:\n{res.answer}\n")
    print("=" * 50)

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("==================================================")
    print("Phase 3 - Semantic Caching Live Test")
    print("==================================================")
    
    t_start = time.time()
    try:
        pipeline = AnswerPipeline()
    except Exception as e:
        print(f"Failed to initialize AnswerPipeline: {e}")
        return
        
    print(f"Pipeline Initialized in {(time.time() - t_start)*1000:.2f} ms\n")
    
    # Ensure cache is clean for the test
    print("Clearing cache before tests...")
    pipeline.cache.clear()
    
    # 1. Exact query - First run (should MISS)
    query_exact = "What was Apple's net sales in 2024?"
    res1 = pipeline.answer(query_exact)
    print_result("1. Initial Query (Expect MISS)", query_exact, res1)
    
    # 1. Exact query - Second run (should HIT exactly)
    res2 = pipeline.answer(query_exact)
    print_result("2. Exact Duplicate Query (Expect HIT, Distance ~0.0)", query_exact, res2)
    
    # 2. Semantic duplicate query (should HIT if distance < 0.15)
    query_semantic = "What were Apple's sales during fiscal year 2024?"
    res3 = pipeline.answer(query_semantic)
    print_result("3. Semantic Duplicate (Expect HIT or close distance)", query_semantic, res3)
    
    # 3. Unrelated query (should MISS)
    query_unrelated = "What are Apple's supply chain risk factors?"
    res4 = pipeline.answer(query_unrelated)
    print_result("4. Unrelated Query (Expect MISS, Distance > threshold)", query_unrelated, res4)
    
    # 4. Cache Clear
    print("\n--- Clearing Cache ---")
    pipeline.cache.clear()
    
    # 5. Query after clear (should MISS)
    res5 = pipeline.answer(query_exact)
    print_result("5. Post-Clear Exact Query (Expect MISS)", query_exact, res5)

if __name__ == "__main__":
    main()
