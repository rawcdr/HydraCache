from qdrant_client import QdrantClient

def main():
    client = QdrantClient(url="http://localhost:6333")
    records, _ = client.scroll(
        collection_name="apple_10k",
        limit=2,
        with_payload=True,
        with_vectors=False
    )
    for r in records:
        print(f"ID: {r.id}")
        print(f"Payload keys: {r.payload.keys()}")
        print(f"Payload: {r.payload}")
        print("-" * 50)

if __name__ == "__main__":
    main()
