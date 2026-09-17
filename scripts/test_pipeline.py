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
    
    t_start_init = time.time()
    try:
        pipeline = PipelineRetriever(pool_size=20, top_k=5)
    except Exception as e:
        print(f"Failed to initialize Pipeline: {e}")
        return
    t_init = (time.time() - t_start_init) * 1000
    print(f"Model Initialization / Download Time: {t_init:.2f} ms\n")
        
    for i, query in enumerate(queries, 1):
        print(f"Query")
        print(f"-----")
        print(f"{query}\n")
        
        t0 = time.time()
        # 1. Hybrid (which contains BM25 + Dense + RRF)
        candidates = pipeline.hybrid_retriever.retrieve_hybrid(query)
        hybrid_time = (time.time() - t0) * 1000
        
        # Print top RRF candidates (simulated pool for output format check)
        print("Hybrid Top-20")
        print("-------------")
        for res in candidates[:3]: # printing top 3 of the 20 to avoid spamming the console
            print(f"{res.chunk_id}")
            print(f"  rrf_score: {res.rrf_score:.4f}")
            print(f"  provenance: {res.provenance}")
        print("  ... (up to 20)\n")
        
        t1 = time.time()
        # 2. Reranker
        try:
            final_results = pipeline.retrieve(query)
        except RuntimeError as e:
            print(f"Pipeline execution failed: {e}")
            break
            
        rerank_time = (time.time() - t1) * 1000
        total_time = (time.time() - t0) * 1000
        
        print("Final Top-5")
        print("-----------")
        for res in final_results:
            print(f"{res.chunk_id}")
            print(f"  rrf_score: {res.rrf_score:.4f}")
            print(f"  rerank_score: {res.rerank_score:.4f}")
            print(f"  page: {res.page_number}")
            print(f"  section: {res.parent_id or 'N/A'}")
            print(f"  chunk_type: {res.chunk_type}")
            
        print(f"\nLatency: Hybrid: {hybrid_time:.2f} ms | Rerank: {rerank_time:.2f} ms | Total: {total_time:.2f} ms\n")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
