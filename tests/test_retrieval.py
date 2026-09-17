import pytest
import os
import urllib.request
from unittest.mock import MagicMock
from qdrant_client.models import QueryResponse
from langchain_core.documents import Document
from src.indexing.sparse import SparseIndexer
from src.retrieval.sparse import SparseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.models import RetrievalResult

# --- Unit Tests ---

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

from unittest.mock import patch

@patch("src.retrieval.dense.QdrantClient")
def test_dense_retrieval_unit(MockQdrantClient):
    # Mock QdrantClient
    mock_client = MockQdrantClient.return_value
    
    # Mock the Qdrant response points using MagicMock
    mock_point_1 = MagicMock()
    mock_point_1.id = "c1"
    mock_point_1.score = 0.92
    mock_point_1.metadata = {"chunk_id": "c1", "document": "Apple makes iPhones.", "page_number": 10, "chunk_type": "text", "parent_id": "p1"}
    
    mock_point_2 = MagicMock()
    mock_point_2.id = "c2"
    mock_point_2.score = 0.75
    mock_point_2.metadata = {"chunk_id": "c2", "document": "Tim Cook is CEO.", "page_number": 11, "chunk_type": "text", "parent_id": "p2"}
    
    mock_client.query.return_value = [mock_point_1, mock_point_2]
    
    retriever = DenseRetriever(collection_name="test_collection")
    results = retriever.retrieve("Who is the CEO?", top_k=2)
    
    # Assert correct types and metadata preservation
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].text == "Apple makes iPhones."
    assert results[0].score == 0.92
    assert results[0].page_number == 10
    assert results[0].parent_id == "p1"
    assert results[0].retrieval_source == "dense"

@patch("src.retrieval.dense.QdrantClient")
def test_dense_retrieval_empty_query(MockQdrantClient):
    mock_client = MockQdrantClient.return_value
    retriever = DenseRetriever()
    
    results = retriever.retrieve("   ", top_k=5)
    assert results == []
    mock_client.query.assert_not_called()

# --- Integration Tests ---

def is_qdrant_running():
    try:
        urllib.request.urlopen("http://localhost:6333/readyz", timeout=1)
        return True
    except Exception:
        return False

@pytest.mark.skipif(not is_qdrant_running(), reason="Qdrant is not running")
def test_dense_retrieval_integration():
    """
    Test against actual local Qdrant (requires apple_10k to be populated from Phase 1).
    """
    retriever = DenseRetriever(collection_name="apple_10k")
    
    results = retriever.retrieve("revenue", top_k=3)
    
    assert len(results) <= 3
    if results:
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score > 0
        assert results[0].text != ""
        assert results[0].retrieval_source == "dense"
        assert "chunk_type" in dir(results[0])
