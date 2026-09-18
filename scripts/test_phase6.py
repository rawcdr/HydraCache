import os
import sys
import uuid
sys.path.append(os.getcwd())

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.indexing.dense import DenseIndexer
from src.indexing.sparse import SparseIndexer
from src.retrieval.answer import AnswerPipeline

def ingest_file(path: str):
    print(f"--- Ingesting {path} ---")
    doc_id = os.path.basename(path).split(".")[0]
    
    parser = PDFParser(path, document_id=doc_id)
    docs = parser.parse()
    
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    
    dense = DenseIndexer()
    dense.index(chunks)
    
    sparse = SparseIndexer()
    sparse.index(chunks)
    print(f"Ingested {len(chunks)} chunks for {doc_id}")

def test_queries():
    pipeline = AnswerPipeline()
    
    # 1. Single hop test
    print("\n--- Testing Single Hop ---")
    q1 = "What was the R&D expense in 2024?"
    res1 = pipeline.answer(q1)
    print(f"Query: {q1}")
    print(f"Mode: {res1.query_mode}")
    print(f"Sub-Qs: {res1.sub_questions}")
    print(f"Answer: {res1.answer}")
    print(f"Latency: {res1.total_latency:.2f}ms")
    print(f"Context chunks: {len(res1.context) if res1.context else 0}")
    
    # 2. Multi hop test
    print("\n--- Testing Multi Hop (Conflict Handling) ---")
    q2 = "What was the reported net sales for 2023, and are there any discrepancies between the 2023 and 2024 reports regarding this figure?"
    res2 = pipeline.answer(q2)
    print(f"Query: {q2}")
    print(f"Mode: {res2.query_mode}")
    print(f"Sub-Qs: {res2.sub_questions}")
    print(f"Answer: {res2.answer}")
    print(f"Latency: {res2.total_latency:.2f}ms")
    print(f"Context chunks: {len(res2.context) if res2.context else 0}")

if __name__ == "__main__":
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    ingest_file("data/raw/apple_10k_2023.pdf")
    ingest_file("data/raw/apple_10k_2024.pdf")
    
    test_queries()
