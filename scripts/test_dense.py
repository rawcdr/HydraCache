import time
from src.retrieval.dense import DenseRetriever

def main():
    queries = [
        "What was Apple's total net sales revenue in 2024?", # Exact financial value
        "What is the role of the App Store?",                 # Terminology/product lookup
        "Describe the impact of macroeconomic conditions.",   # Narrative question
        "Show me the net sales by reportable segment.",       # Table-oriented question
        "What are the risk factors related to supply chain?"  # Section-specific question
    ]
    
    print("==================================================")
    print("Phase 2 - Milestone 2: Dense Retrieval Manual Test")
    print("==================================================")
    
    try:
        retriever = DenseRetriever(collection_name="apple_10k")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return
        
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: {query}")
        
        start_time = time.time()
        results = retriever.retrieve(query, top_k=3)
        latency = (time.time() - start_time) * 1000
        
        print(f"    Latency: {latency:.2f} ms")
        print(f"    Top-k returned: {len(results)}")
        
        for j, res in enumerate(results, 1):
            print(f"    Result {j}:")
            print(f"      - Chunk ID: {res.chunk_id}")
            print(f"      - Score: {res.score:.4f}")
            print(f"      - Page: {res.page_number}")
            print(f"      - Chunk Type: {res.chunk_type}")
            snippet = res.text.replace("\n", " ")[:100] + "..." if len(res.text) > 100 else res.text.replace("\n", " ")
            print(f"      - Text Snippet: {snippet}")

if __name__ == "__main__":
    main()
