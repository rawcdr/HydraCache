import pytest
import os
from langchain_core.documents import Document
from src.indexing.sparse import SparseIndexer
from src.retrieval.sparse import SparseRetriever
from src.retrieval.models import RetrievalResult

def test_bm25_retrieval(tmp_path):
    # 1. Build a mock index
    index_path = os.path.join(tmp_path, "bm25_test.pkl")
    indexer = SparseIndexer(save_path=index_path)
    
    docs = [
        Document(page_content="Apple is a global tech company.", metadata={"chunk_id": "c1", "source_file": "doc1.pdf", "page_number": 1, "chunk_type": "text"}),
        Document(page_content="The iPhone is their most popular product.", metadata={"chunk_id": "c2", "source_file": "doc1.pdf", "page_number": 2, "chunk_type": "text"}),
        Document(page_content="Microsoft makes Windows.", metadata={"chunk_id": "c3", "source_file": "doc2.pdf", "page_number": 1, "chunk_type": "text"})
    ]
    indexer.index(docs)
    
    # 2. Test retrieval
    retriever = SparseRetriever(index_path=index_path)
    results = retriever.retrieve("Apple iPhone", top_k=5)
    
    assert len(results) == 2  # Only two docs have the keywords
    assert isinstance(results[0], RetrievalResult)
    assert results[0].retrieval_source == "bm25"
    
    # Check that chunk_ids are c1 and c2 in some order
    retrieved_ids = {r.chunk_id for r in results}
    assert retrieved_ids == {"c1", "c2"}
