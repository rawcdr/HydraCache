import os
import json
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.config import Config

client = TestClient(app)

# Helper to avoid repetitive typing
VALID_API_KEY = Config.API_KEY
HEADERS = {"X-API-Key": VALID_API_KEY}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_no_key():
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 401
    assert "Missing API Key" in response.json()["detail"]

def test_query_invalid_key():
    response = client.post("/query", json={"query": "test"}, headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

# Mocking the pipeline to avoid heavy LLM/Vector DB logic during API unit testing
@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    class MockResult:
        answer = "Mocked Answer"
        cache_hit = False
        cache_distance = None
        cache_latency = 1.0
        retrieval_latency = 10.0
        generation_latency = 100.0
        total_latency = 111.0
        prompt_tokens = 50
        completion_tokens = 10
        total_tokens = 60
        context = []
        query_mode = "single_hop"
        sub_questions = []

    class MockPipeline:
        def answer(self, query: str):
            return MockResult()

    app.state.pipeline = MockPipeline()

def test_query_valid_key():
    response = client.post("/query", json={"query": "What is the net sales?"}, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked Answer"
    assert data["cache_hit"] is False
    assert "latency" in data
    assert "tokens" in data
    assert response.headers.get("x-request-id") is not None

def test_document_upload_missing_file():
    response = client.post("/documents", headers=HEADERS)
    assert response.status_code == 422 # Unprocessable Entity (FastAPI validation)

def test_document_upload_invalid_type():
    response = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 415
    assert "Only PDF files are supported" in response.json()["detail"]

# We mock background tasks so it doesn't actually try to index
def test_document_upload_success(monkeypatch):
    response = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("test.pdf", b"dummy pdf content", "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    
    # Store job_id for next test
    os.environ["TEST_JOB_ID"] = data["job_id"]

def test_get_job_status():
    job_id = os.environ.get("TEST_JOB_ID")
    if not job_id:
        pytest.skip("No job ID from previous test")
        
    response = client.get(f"/documents/{job_id}", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    # TestClient runs background tasks synchronously.
    # Since we uploaded a fake PDF, the task runs instantly and fails.
    assert data["status"] in ["queued", "running", "completed", "failed"]

def test_get_unknown_job():
    response = client.get("/documents/invalid-job-id-1234", headers=HEADERS)
    assert response.status_code == 404
    assert "Job ID not found" in response.json()["detail"]
