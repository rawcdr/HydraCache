import os
import sys
import json
import time
import re
import math
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import pandas as pd

from eval.baseline import BaselineSystem
from src.retrieval.pipeline import PipelineRetriever
from src.retrieval.answer import AnswerPipeline
from src.cache.semantic_cache import SemanticCache

RESULTS_PATH = "eval/results.json"
MAX_SLEEP_SECONDS = 300  # Do not wait more than 5 minutes for a single TPD limit block

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

def parse_groq_wait_time(error_str: str) -> float:
    m = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)', error_str)
    if m:
        h = float(m.group(1) or 0)
        mins = float(m.group(2) or 0)
        s = float(m.group(3) or 0)
        return h * 3600 + mins * 60 + s
    return 60.0 # Default fallback if parsing fails

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
        
        # 1. GENERATION PHASE
        res = answer_func(question)
        
        status = "SUCCESS"
        error_type = None
        
        # Check if the pipeline caught an error during generation
        if str(res.answer).startswith("Generation failed"):
            if "429" in str(res.answer) or "Rate limit" in str(res.answer) or "rate_limit_exceeded" in str(res.answer):
                status = "EVALUATOR_RATE_LIMIT"
                error_type = "Synthesis Rate Limit"
            else:
                status = "SYSTEM_ERROR"
                error_type = "Synthesis Failure"
                
        # Base result
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
            "faithfulness": float('nan'),
            "answer_relevancy": float('nan'),
            "context_precision": float('nan'),
            "context_recall": float('nan'),
            "status": status,
            "error_type": error_type
        }
        
        # 2. EVALUATION PHASE (Only if generation succeeded)
        if status == "SUCCESS":
            ragas_data = {
                "question": [question],
                "answer": [res.answer],
                "contexts": [[c.text for c in (res.context or [])]],
                "ground_truth": [qa["ground_truth_answer"]]
            }
            dataset = Dataset.from_dict(ragas_data)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"  -> Running Ragas metrics (Attempt {attempt+1}/{max_retries})...")
                    # We set raise_exceptions=True so we can catch and explicitly handle RateLimits
                    evaluation = evaluate(
                        dataset=dataset,
                        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                        llm=chat_model,
                        embeddings=embeddings,
                        is_async=False,
                        run_config=run_config,
                        raise_exceptions=True 
                    )
                    
                    eval_df = evaluation.to_pandas()
                    result_record["faithfulness"] = float(eval_df.iloc[0]["faithfulness"]) if "faithfulness" in eval_df and pd.notna(eval_df.iloc[0]["faithfulness"]) else float('nan')
                    result_record["answer_relevancy"] = float(eval_df.iloc[0]["answer_relevancy"]) if "answer_relevancy" in eval_df and pd.notna(eval_df.iloc[0]["answer_relevancy"]) else float('nan')
                    result_record["context_precision"] = float(eval_df.iloc[0]["context_precision"]) if "context_precision" in eval_df and pd.notna(eval_df.iloc[0]["context_precision"]) else float('nan')
                    result_record["context_recall"] = float(eval_df.iloc[0]["context_recall"]) if "context_recall" in eval_df and pd.notna(eval_df.iloc[0]["context_recall"]) else float('nan')
                    
                    # If any metric is NaN, Ragas internal logic failed without raising an exception (e.g. malformed generation)
                    if math.isnan(result_record["faithfulness"]) or math.isnan(result_record["answer_relevancy"]):
                         result_record["status"] = "EVALUATOR_API_ERROR"
                         result_record["error_type"] = "Ragas returned NaN for metrics"
                         
                    break # Success!
                    
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit_exceeded" in err_str:
                        wait_time = parse_groq_wait_time(err_str)
                        if wait_time > MAX_SLEEP_SECONDS:
                            print(f"  -> Rate limit block is too long ({wait_time}s). Aborting evaluation for this row.")
                            result_record["status"] = "EVALUATOR_RATE_LIMIT"
                            result_record["error_type"] = "TPD Limit Exhausted"
                            break
                        else:
                            sleep_time = wait_time + 2.0
                            print(f"  -> Rate limit hit. Sleeping for {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                            if attempt == max_retries - 1:
                                result_record["status"] = "EVALUATOR_RATE_LIMIT"
                                result_record["error_type"] = "Max Retries Exceeded"
                    else:
                        print(f"  -> Unexpected Ragas Error: {err_str}")
                        result_record["status"] = "EVALUATOR_API_ERROR"
                        result_record["error_type"] = "Exception during Ragas execution"
                        break
        
        results.append(result_record)
        save_results(results)
        print(f"  -> Saved {q_id} for {system_name}. Status: {result_record['status']}")
        
        # Sleep slightly to further mitigate RPM limits
        if result_record["status"] == "SUCCESS":
            time.sleep(3)

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
        max_retries=1  # We handle retries manually for big backoffs
    )
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    run_config = RunConfig(max_workers=1, max_retries=1, max_wait=60)
    
    results = load_existing_results()
    if not isinstance(results, list):
        save_results([])
        
    evaluate_system_incrementally("Baseline", qa_pairs, baseline.answer, chat_model, embeddings, run_config)
    evaluate_system_incrementally("Optimized", qa_pairs, optimized_answer_func, chat_model, embeddings, run_config)
    
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()
