import pytest
from unittest.mock import patch, MagicMock
from src.cache.semantic_cache import SemanticCache, CacheResult
from src.retrieval.answer import AnswerPipeline, AnswerResult, generate_answer
from src.retrieval.models import RetrievalResult
import numpy as np
from redis.exceptions import ResponseError

@pytest.fixture
def mock_embedding():
    with patch('src.cache.semantic_cache.TextEmbedding') as mock_embed:
        # Mock embedding model to return a fixed 384-dim array
        mock_instance = mock_embed.return_value
        mock_instance.embed.return_value = [np.zeros(384, dtype="float32")]
        yield mock_instance

@pytest.fixture
def mock_redis():
    with patch('src.cache.semantic_cache.Redis') as mock_r:
        mock_instance = mock_r.from_url.return_value
        # Default mock index exists
        mock_instance.ft.return_value.info.return_value = {"index_name": "idx:semantic_cache"}
        yield mock_instance

def test_cache_initialization(mock_embedding, mock_redis):
    cache = SemanticCache()
    assert cache.redis is not None
    mock_redis.ft.assert_called()

def test_index_creation_when_not_exists(mock_embedding, mock_redis):
    # Simulate "Unknown Index name"
    mock_redis.ft.return_value.info.side_effect = ResponseError("Unknown Index name")
    cache = SemanticCache()
    mock_redis.ft.return_value.create_index.assert_called_once()

def test_redis_failure_behavior(mock_embedding):
    with patch('src.cache.semantic_cache.Redis') as mock_r:
        mock_r.from_url.side_effect = Exception("Connection Refused")
        cache = SemanticCache()
        assert cache.redis is None
        
        # Test fail-open gracefully
        res = cache.lookup("test")
        assert res.hit is False
        assert cache.store("test", "ans") is False
        cache.clear() # Should not crash

def test_cache_store(mock_embedding, mock_redis):
    cache = SemanticCache()
    success = cache.store("question", "answer text")
    assert success is True
    mock_redis.hset.assert_called_once()
    mock_redis.expire.assert_called_once()

def test_cache_store_empty_answer(mock_embedding, mock_redis):
    cache = SemanticCache()
    assert cache.store("question", "") is False
    assert cache.store("question", "   ") is False
    mock_redis.hset.assert_not_called()

def test_cache_lookup_hit(mock_embedding, mock_redis):
    cache = SemanticCache(threshold=0.15)
    
    mock_doc = MagicMock()
    mock_doc.distance = "0.05"
    mock_doc.answer = "cached answer"
    mock_doc.id = "cache:123"
    
    mock_redis.ft.return_value.search.return_value.docs = [mock_doc]
    
    res = cache.lookup("test query")
    assert res.hit is True
    assert res.answer == "cached answer"
    assert res.distance == 0.05
    assert res.cache_id == "cache:123"

def test_cache_lookup_miss_threshold(mock_embedding, mock_redis):
    cache = SemanticCache(threshold=0.15)
    
    mock_doc = MagicMock()
    mock_doc.distance = "0.20" # Above threshold
    
    mock_redis.ft.return_value.search.return_value.docs = [mock_doc]
    
    res = cache.lookup("test query")
    assert res.hit is False
    assert res.distance == 0.20
    assert res.answer is None

def test_cache_clear(mock_embedding, mock_redis):
    cache = SemanticCache()
    mock_redis.keys.return_value = ["cache:1", "cache:2"]
    cache.clear()
    mock_redis.delete.assert_called_once_with("cache:1", "cache:2")

# --- AnswerPipeline Orchestration Tests ---

@patch("src.retrieval.answer.generate_answer")
def test_pipeline_miss_path(mock_generate):
    pipeline_mock = MagicMock()
    cache_mock = MagicMock()
    
    # Simulate a MISS
    cache_mock.lookup.return_value = CacheResult(hit=False, distance=0.2)
    pipeline_mock.retrieve.return_value = [RetrievalResult(chunk_id="1", text="txt", score=1.0, source_file="x", page_number=1, chunk_type="txt")]
    
    mock_generate.return_value = {
        "answer": "Generated answer based on context.",
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20
    }
    
    answer_pipeline = AnswerPipeline(pipeline=pipeline_mock, cache=cache_mock)
    res = answer_pipeline.answer("new query")
    
    assert res.cache_hit is False
    assert res.answer == "Generated answer based on context."
    assert res.total_tokens == 20
    pipeline_mock.retrieve.assert_called_once_with("new query")
    cache_mock.store.assert_called_once()

def test_pipeline_hit_path():
    pipeline_mock = MagicMock()
    cache_mock = MagicMock()
    
    # Simulate a HIT
    cache_mock.lookup.return_value = CacheResult(hit=True, answer="I am cached", distance=0.01)
    
    answer_pipeline = AnswerPipeline(pipeline=pipeline_mock, cache=cache_mock)
    res = answer_pipeline.answer("cached query")
    
    assert res.cache_hit is True
    assert res.answer == "I am cached"
    
    # Ensure retrieval and store were bypassed
    pipeline_mock.retrieve.assert_not_called()
    cache_mock.store.assert_not_called()
