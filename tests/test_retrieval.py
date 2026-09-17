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

# --- RRF and Hybrid Tests ---

def test_rrf_fuse_mathematics():
    from src.retrieval.fusion import rrf_fuse
    
    # 2 dummy lists with 2 overlapping chunks and 1 unique each
    r1 = RetrievalResult(chunk_id="c1", text="text1", score=1.0, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list1")
    r2 = RetrievalResult(chunk_id="c2", text="text2", score=0.9, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list1")
    r3 = RetrievalResult(chunk_id="c3", text="text3", score=0.8, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list1")
    
    list_a = [r1, r2, r3]  # c1 is rank 1, c2 is rank 2, c3 is rank 3
    
    # list b: c2 is rank 1, c1 is rank 2, c4 is rank 3
    r1_b = RetrievalResult(chunk_id="c1", text="text1", score=0.8, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list2")
    r2_b = RetrievalResult(chunk_id="c2", text="text2", score=0.9, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list2")
    r4 = RetrievalResult(chunk_id="c4", text="text4", score=0.99, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="list2")
    list_b = [r2_b, r1_b, r4]
    
    k = 60
    # Expected scores:
    # c1 = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.016129 = 0.032522
    # c2 = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522
    # c3 = 1/(60+3) = 1/63 = 0.015873
    # c4 = 1/(60+3) = 1/63 = 0.015873
    
    fused = rrf_fuse([list_a, list_b], k=k, top_k=20)
    
    assert len(fused) == 4
    assert set(fused[0].provenance) == {"list1", "list2"}
    assert set(fused[1].provenance) == {"list1", "list2"}
    assert fused[0].retrieval_source == "hybrid"
    
    # Check descending order by rrf_score
    assert fused[0].rrf_score >= fused[1].rrf_score
    assert fused[1].rrf_score >= fused[2].rrf_score
    assert fused[2].rrf_score >= fused[3].rrf_score

@patch("src.retrieval.dense.DenseRetriever.retrieve")
@patch("src.retrieval.sparse.SparseRetriever.retrieve")
def test_hybrid_retrieval_orchestration(mock_sparse, mock_dense):
    from src.retrieval.hybrid import HybridRetriever
    
    r1 = RetrievalResult(chunk_id="c1", text="text1", score=1.0, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="bm25")
    r2 = RetrievalResult(chunk_id="c2", text="text2", score=0.9, source_file="doc1.pdf", page_number=1, chunk_type="text", retrieval_source="dense")
    
    mock_sparse.return_value = [r1]
    mock_dense.return_value = [r2]
    
    # We patch the inner components but we need to mock their initialization
    # Actually, we can just patch the classes or pass dependencies if we used DI
    # But since they're hardcoded in __init__, we just mock the retrieve methods of the instances created.
    with patch("src.retrieval.sparse.SparseRetriever.__init__", return_value=None), \
         patch("src.retrieval.dense.DenseRetriever.__init__", return_value=None):
             
        retriever = HybridRetriever(pool_size=1)
        # Manually attach mocks because __init__ bypassed
        retriever.sparse_retriever.retrieve = mock_sparse
        retriever.dense_retriever.retrieve = mock_dense
        
        results = retriever.retrieve_hybrid("test query")
        
        assert len(results) == 1  # pool_size is 1
        assert results[0].chunk_id in ["c1", "c2"]
        mock_sparse.assert_called_once_with("test query", 20)
        mock_dense.assert_called_once_with("test query", 20)

# --- Reranker & Pipeline Tests ---

@patch("src.retrieval.reranker.TextCrossEncoder")
def test_reranker_logic(MockCrossEncoder):
    from src.retrieval.reranker import Reranker
    
    mock_model = MockCrossEncoder.return_value
    # Reranker returns an iterable of floats in the same order as candidates
    mock_model.rerank.return_value = iter([0.1, 0.9, 0.5])
    
    r1 = RetrievalResult(chunk_id="c1", text="text1", score=1.0, source_file="doc1.pdf", page_number=1, chunk_type="text", rrf_score=0.03, retrieval_source="hybrid")
    r2 = RetrievalResult(chunk_id="c2", text="text2", score=0.9, source_file="doc1.pdf", page_number=1, chunk_type="text", rrf_score=0.02, retrieval_source="hybrid")
    r3 = RetrievalResult(chunk_id="c3", text="text3", score=0.8, source_file="doc1.pdf", page_number=1, chunk_type="text", rrf_score=0.01, retrieval_source="hybrid")
    
    reranker = Reranker()
    reranked = reranker.rerank("query", [r1, r2, r3], top_k=2)
    
    assert len(reranked) == 2
    # c2 got 0.9, c3 got 0.5, c1 got 0.1
    # Sorting by rerank_score descending -> c2, c3
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rerank_score == 0.9
    assert reranked[0].rrf_score == 0.02
    assert reranked[1].chunk_id == "c3"
    assert reranked[1].rerank_score == 0.5
    
    mock_model.rerank.assert_called_once_with("query", ["text1", "text2", "text3"])

@patch("src.retrieval.pipeline.HybridRetriever")
@patch("src.retrieval.pipeline.Reranker")
def test_pipeline_orchestration(MockReranker, MockHybrid):
    from src.retrieval.pipeline import PipelineRetriever
    
    mock_hybrid_instance = MockHybrid.return_value
    mock_reranker_instance = MockReranker.return_value
    
    r1 = RetrievalResult(chunk_id="c1", text="text1", score=1.0, source_file="doc1.pdf", page_number=1, chunk_type="text", rrf_score=0.03)
    mock_hybrid_instance.retrieve_hybrid.return_value = [r1]
    
    r1_reranked = RetrievalResult(chunk_id="c1", text="text1", score=1.0, source_file="doc1.pdf", page_number=1, chunk_type="text", rrf_score=0.03, rerank_score=0.99)
    mock_reranker_instance.rerank.return_value = [r1_reranked]
    
    pipeline = PipelineRetriever(top_k=5)
    results = pipeline.retrieve("test query")
    
    assert len(results) == 1
    assert results[0].rerank_score == 0.99
    
    mock_hybrid_instance.retrieve_hybrid.assert_called_once_with("test query")
    mock_reranker_instance.rerank.assert_called_once_with("test query", [r1], top_k=5)

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
