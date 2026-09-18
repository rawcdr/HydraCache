import json
from qdrant_client import QdrantClient

def main():
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "apple_10k"
    
    # Scroll through some chunks
    records, next_page = client.scroll(
        collection_name=collection_name,
        limit=50,
        with_payload=True,
        with_vectors=False
    )
    
    samples = []
    for r in records:
        samples.append({
            "text": r.payload.get("document", ""),
            "page": r.payload.get("page_number", ""),
            "section": r.payload.get("parent_id", ""),
            "chunk_type": r.payload.get("chunk_type", "")
        })
        
    with open("scratch/sample_chunks.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("scratch", exist_ok=True)
    main()
