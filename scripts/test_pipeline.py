import time
import sys
from src.retrieval.pipeline import PipelineRetriever

def main():
    queries = [
        "What was Apple's total net sales revenue in 2024?",
        "What is the role of the App Store?",
        "Describe the impact of macroeconomic conditions.",
        "Show me the net sales by reportable segment.",
        "What are the risk factors related to supply chain?"
    ]
    
    print("==================================================")
    print("Phase 2 - Milestone 4: End-to-End Pipeline")
    print("==================================================")
    
    try:
        # Pipeline orchestrates Hybrid (BM25+Dense) -> RRF -> Reranking
        pipeline = PipelineRetriever(pool_size=20, top_k=5)
    except Exception as e:
        print(f"Failed to initialize Pipeline: {e}")
        return
        
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: '{query}'")
        print("-" * 50)
        
        # In a real environment, we would measure latency internally or patch the components.
        # For simplicity here, we rely on the overall pipeline latency, but since the prompt requested
        # latency breakdown, we'll manually invoke the steps here for tracing, OR we can just time the whole thing.
        # Let's just do a manual trace for this script.
        
        t0 = time.time()
        # 1. Hybrid (which contains BM25 + Dense + RRF)
        candidates = pipeline.hybrid_retriever.retrieve_hybrid(query)
        hybrid_time = (time.time() - t0) * 1000
        
        # Print top RRF candidates
        print(f"RRF Top-20 (showing top 3):")
        for res in candidates[:3]:
            print(f"    - {res.chunk_id} | RRF Score: {res.rrf_score:.4f} | Page: {res.page_number} | Section: {res.chunk_type}")
        
        t1 = time.time()
        # 2. Reranker
        if pipeline.reranker is None:
            print("Reranker failed to load. Skipping reranking step.")
            continue
            
        final_results = pipeline.reranker.rerank(query, candidates, top_k=5)
        rerank_time = (time.time() - t1) * 1000
        total_time = (time.time() - t0) * 1000
        
        print(f"\nReranked Top-5:")
        for res in final_results:
            print(f"    - {res.chunk_id} | Rerank Score: {res.rerank_score:.4f} | RRF Score: {res.rrf_score:.4f} | Page: {res.page_number} | Type: {res.chunk_type}")
            
        print(f"\nLatency: Hybrid: {hybrid_time:.2f} ms | Rerank: {rerank_time:.2f} ms | Total: {total_time:.2f} ms\n")

if __name__ == "__main__":
    main()
