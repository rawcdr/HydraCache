import os
import sys
import json
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv

from eval.baseline import BaselineSystem
from src.retrieval.pipeline import PipelineRetriever
from src.retrieval.answer import AnswerPipeline
from src.cache.semantic_cache import SemanticCache

RESULTS_PATH = "eval/results.json"

def load_existing_results() -> List[Dict[str, Any]]:
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_results(results: List[Dict[str, Any]]):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def evaluate_system_incrementally(system_name: str, qa_pairs: List[Dict], answer_func, chat_model, embeddings, run_config):
    print(f"\n--- Evaluating {system_name} Incrementally ---")
    
    results = load_existing_results()
    evaluated_ids = {r["question_id"] for r in results if r.get("system") == system_name}
    
    for i, qa in enumerate(qa_pairs):
        q_id = qa["id"]
        if q_id in evaluated_ids:
            print(f"[{i+1}/{len(qa_pairs)}] Skipping {q_id} - Already evaluated.")
            continue
            
        question = qa["question"]
        print(f"[{i+1}/{len(qa_pairs)}] Processing {q_id}: {question}")
        
        # Invoke system
        res = answer_func(question)
        
        # Ragas evaluation inputs
        ragas_data = {
            "question": [question],
            "answer": [res.answer],
            "contexts": [[c.text for c in (res.context or [])]],
            "ground_truth": [qa["ground_truth_answer"]]
        }
        dataset = Dataset.from_dict(ragas_data)
        
        print(f"  -> Running Ragas metrics...")
        evaluation = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=chat_model,
            embeddings=embeddings,
            is_async=False,
            run_config=run_config,
            raise_exceptions=False
        )
        
        eval_df = evaluation.to_pandas()
        
        # Build raw result
        result_record = {
            "question_id": q_id,
            "category": qa["category"],
            "system": system_name,
            "question": question,
            "answer": res.answer,
            "ground_truth": qa["ground_truth_answer"],
            "retrieved_chunks": [c.text for c in (res.context or [])],
            "latency": res.total_latency,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.total_tokens,
            "faithfulness": float(eval_df.iloc[0]["faithfulness"]) if "faithfulness" in eval_df else 0.0,
            "answer_relevancy": float(eval_df.iloc[0]["answer_relevancy"]) if "answer_relevancy" in eval_df else 0.0,
            "context_precision": float(eval_df.iloc[0]["context_precision"]) if "context_precision" in eval_df else 0.0,
            "context_recall": float(eval_df.iloc[0]["context_recall"]) if "context_recall" in eval_df else 0.0
        }
        
        results.append(result_record)
        save_results(results)
        print(f"  -> Saved {q_id} for {system_name}.")
        
        # Sleep slightly to further mitigate rate limits
        time.sleep(2)

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    load_dotenv()
    
    if not os.path.exists("eval/test_set.json"):
        raise FileNotFoundError("eval/test_set.json not found")
        
    with open("eval/test_set.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    baseline = BaselineSystem()
    pipeline = PipelineRetriever()
    cache = SemanticCache()
    
    def optimized_answer_func(query: str):
        cache.clear() # Force miss for pure generation testing
        return AnswerPipeline(pipeline=pipeline, cache=cache).answer(query)
        
    chat_model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
        max_retries=5
    )
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Restrict concurrency to 1 and enable internal retries in Ragas
    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=60)
    
    # Initialize empty results file if missing or if it's the old dummy array
    results = load_existing_results()
    if not isinstance(results, list):
        save_results([])
        
    evaluate_system_incrementally("Baseline", qa_pairs, baseline.answer, chat_model, embeddings, run_config)
    evaluate_system_incrementally("Optimized", qa_pairs, optimized_answer_func, chat_model, embeddings, run_config)
    
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()
