import time
import sys
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.dense import DenseRetriever

def main():
    queries = [
        "What was Apple's total net sales revenue in 2024?",
        "What is the role of the App Store?",
        "Describe the impact of macroeconomic conditions.",
        "Show me the net sales by reportable segment.",
        "What are the risk factors related to supply chain?"
    ]
    
    print("==================================================")
    print("Phase 2 - Milestone 3: Parallel Retrieval + RRF")
    print("==================================================")
    
    try:
        hybrid_retriever = HybridRetriever(pool_size=20)
        # We also create direct retrievers just to measure individual latency for comparison
        sparse_retriever = SparseRetriever()
        dense_retriever = DenseRetriever()
    except Exception as e:
        print(f"Failed to initialize retrievers: {e}")
        return
        
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: '{query}'")
        print("-" * 50)
        
        # 1. BM25 baseline
        t0 = time.time()
        res_sparse = sparse_retriever.retrieve(query, top_k=20)
        t_bm25 = (time.time() - t0) * 1000
        
        # 2. Dense baseline
        t0 = time.time()
        res_dense = dense_retriever.retrieve(query, top_k=20)
        t_dense = (time.time() - t0) * 1000
        
        # 3. Hybrid Parallel Execution
        t0 = time.time()
        res_hybrid = hybrid_retriever.retrieve_hybrid(query)
        t_hybrid = (time.time() - t0) * 1000
        
        print(f"BM25 latency:   {t_bm25:.2f} ms")
        print(f"Dense latency:  {t_dense:.2f} ms")
        print(f"Hybrid latency: {t_hybrid:.2f} ms (Parallel)")
        print(f"Total top-20 candidates: {len(res_hybrid)}")
        
        # Show top 3 fused results
        print("\nTop 3 Fused Results:")
        for j, res in enumerate(res_hybrid[:3], 1):
            print(f"    {j}. Chunk ID: {res.chunk_id}")
            print(f"       RRF Score: {res.rrf_score:.4f}")
            print(f"       Provenance: {res.provenance}")
            print(f"       Page: {res.page_number} | Section: {res.chunk_type}")

if __name__ == "__main__":
    main()
