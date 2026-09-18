import pytest
import os
from unittest.mock import patch, MagicMock
from src.generation.synthesize import generate_answer
from src.retrieval.models import RetrievalResult

@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        yield

def test_generate_answer_empty_context(mock_env):
    res = generate_answer("test query", [])
    assert res["answer"] == "I could not find an answer in the provided documents."
    assert res["total_tokens"] == 0

@patch("src.generation.synthesize.requests.post")
def test_generate_answer_success(mock_post, mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is a grounded answer citing [Page: 1]." }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    mock_post.return_value = mock_response
    
    chunks = [
        RetrievalResult(chunk_id="1", text="some context", score=0.9, source_file="doc1", page_number=1, chunk_type="text")
    ]
    
    res = generate_answer("What is the context?", chunks)
    
    assert res["answer"] == "This is a grounded answer citing [Page: 1]."
    assert res["total_tokens"] == 15
    
    # Verify the payload structure
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "openai/gpt-oss-20b"
    assert "some context" in payload["messages"][1]["content"]

@patch("src.generation.synthesize.requests.post")
def test_generate_answer_failure(mock_post, mock_env):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response
    
    chunks = [RetrievalResult(chunk_id="1", text="x", score=0.9, source_file="doc", page_number=1, chunk_type="text")]
    
    with pytest.raises(RuntimeError, match="LLM Generation failed with status code 500"):
        generate_answer("test?", chunks)
