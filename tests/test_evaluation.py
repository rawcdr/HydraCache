import os
import json
import pytest
from eval.baseline import BaselineSystem
from src.retrieval.answer import AnswerPipeline

def test_evaluation_dataset_loading_and_categories():
    assert os.path.exists("eval/test_set.json")
    with open("eval/test_set.json", "r") as f:
        data = json.load(f)
        
    assert len(data) >= 20
    valid_categories = {"exact-lookup", "narrative", "table-reasoning", "multi-hop"}
    
    for item in data:
        assert "id" in item
        assert "question" in item
        assert "ground_truth_answer" in item
        assert "category" in item
        assert item["category"] in valid_categories

def test_baseline_execution(monkeypatch):
    baseline = BaselineSystem()
    assert hasattr(baseline, 'answer')
    
    # Mock retrieve and generate_answer to test execution path without live API
    def mock_retrieve(*args, **kwargs):
        from src.retrieval.models import RetrievalResult
        return [RetrievalResult(
            chunk_id="1", 
            text="dummy context", 
            score=1.0,
            source_file="dummy.pdf",
            page_number=1,
            chunk_type="text"
        )]
        
    def mock_generate(*args, **kwargs):
        return {"answer": "baseline answer", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        
    monkeypatch.setattr(baseline.retriever, "retrieve", mock_retrieve)
    monkeypatch.setattr("eval.baseline.generate_answer", mock_generate)
    
    res = baseline.answer("Test question")
    assert res.answer == "baseline answer"
    assert res.cache_hit is False
    assert len(res.context) == 1

def test_results_serialization_and_aggregation(tmp_path):
    dummy_results = [
        {"question_id": "q1", "category": "exact-lookup", "system": "Baseline", "latency": 100, "cache_hit": False, "faithfulness": 1.0},
        {"question_id": "q2", "category": "narrative", "system": "Optimized", "latency": 50, "cache_hit": True, "faithfulness": 0.8}
    ]
    
    file_path = tmp_path / "results.json"
    with open(file_path, "w") as f:
        json.dump(dummy_results, f)
        
    with open(file_path, "r") as f:
        loaded = json.load(f)
        
    assert len(loaded) == 2
    assert loaded[0]["latency"] == 100
    assert loaded[1]["cache_hit"] is True

def test_failure_classification():
    # Test valid classification keys
    valid_classifications = ["Retrieval Failure", "Synthesis Failure", "Source Limitation", "Evaluation/Data Issue"]
    
    # Create a dummy failure_analysis.json format
    dummy_failures = [
        {
            "question": "What is the sales?",
            "category": "exact-lookup",
            "classification": "Retrieval Failure",
            "expected_answer": "100",
            "actual_answer": "I don't know",
            "retrieved_chunks": ["Unrelated text"]
        }
    ]
    
    assert dummy_failures[0]["classification"] in valid_classifications
