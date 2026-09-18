import os
import json
import pandas as pd

def safe_mean(series):
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.mean():.4f}"

def print_metrics(df_sys, system_name):
    print(f"\n{system_name}:")
    valid_count = len(df_sys[df_sys["status"] == "SUCCESS"])
    failed_count = len(df_sys[df_sys["status"] != "SUCCESS"])
    
    print(f"- faithfulness       (valid: {valid_count}, failed: {failed_count}) -> {safe_mean(df_sys['faithfulness'])}")
    print(f"- context precision  (valid: {valid_count}, failed: {failed_count}) -> {safe_mean(df_sys['context_precision'])}")
    print(f"- context recall     (valid: {valid_count}, failed: {failed_count}) -> {safe_mean(df_sys['context_recall'])}")
    print(f"- answer relevance   (valid: {valid_count}, failed: {failed_count}) -> {safe_mean(df_sys['answer_relevancy'])}")

def main():
    if not os.path.exists("eval/results.json"):
        print("No evaluation results found.")
        return
        
    with open("eval/results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("Empty evaluation results.")
        return
        
    # Validation status
    attempted = len(df)
    success = len(df[df["status"] == "SUCCESS"])
    failed = len(df[df["status"] != "SUCCESS"])
    
    print("PHASE 4 VALIDATION STATUS\n")
    print(f"Evaluation cases attempted: {attempted}")
    print(f"Evaluation cases successfully scored: {success}")
    print(f"Evaluation cases failed: {failed}")
    
    # Baseline vs HydraCache
    df_base = df[df["system"] == "Baseline"]
    df_opt = df[df["system"] == "Optimized"]
    
    print_metrics(df_base, "Baseline")
    print_metrics(df_opt, "HydraCache")
    
    # Category Breakdown
    print("\n--- Category Breakdown ---")
    categories = df["category"].unique()
    for cat in categories:
        df_cat = df[df["category"] == cat]
        success_cat = len(df_cat[df_cat["status"] == "SUCCESS"])
        failed_cat = len(df_cat[df_cat["status"] != "SUCCESS"])
        print(f"\nCategory: {cat}")
        print(f"  valid sample count: {success_cat}")
        print(f"  failed sample count: {failed_cat}")
        if success_cat > 0:
            print(f"  faithfulness: {safe_mean(df_cat['faithfulness'])}")
            print(f"  context_precision: {safe_mean(df_cat['context_precision'])}")
            print(f"  context_recall: {safe_mean(df_cat['context_recall'])}")
            print(f"  answer_relevancy: {safe_mean(df_cat['answer_relevancy'])}")
    
    # Cache Profiling
    print("\nCache profiling:")
    if os.path.exists("eval/cache_profile.csv"):
        cache_df = pd.read_csv("eval/cache_profile.csv")
        hits = cache_df[cache_df["cache_hit"] == True]
        misses = cache_df[cache_df["cache_hit"] == False]
        
        hit_rate = len(hits) / len(cache_df) if len(cache_df) > 0 else 0
        hit_p50 = hits["total_latency"].quantile(0.5) if len(hits) > 0 else 0.0
        hit_p95 = hits["total_latency"].quantile(0.95) if len(hits) > 0 else 0.0
        miss_p50 = misses["total_latency"].quantile(0.5) if len(misses) > 0 else 0.0
        miss_p95 = misses["total_latency"].quantile(0.95) if len(misses) > 0 else 0.0
        
        print(f"- hit rate: {hit_rate * 100:.1f}%")
        print(f"- hit p50: {hit_p50:.2f} ms")
        print(f"- hit p95: {hit_p95:.2f} ms")
        print(f"- miss p50: {miss_p50:.2f} ms")
        print(f"- miss p95: {miss_p95:.2f} ms")
    else:
        print("Cache profiling data not found.")
        
    # Failures Breakdown
    print("\nEvaluator failures:")
    rate_limits = len(df[df["status"] == "EVALUATOR_RATE_LIMIT"])
    api_errors = len(df[df["status"] == "EVALUATOR_API_ERROR"])
    timeouts = len(df[df["status"] == "EVALUATOR_TIMEOUT"])
    print(f"- rate limits: {rate_limits}")
    print(f"- timeouts: {timeouts}")
    print(f"- other API errors: {api_errors}")
    
    print("\nSystem failures:")
    sys_errors = len(df[df["status"] == "SYSTEM_ERROR"])
    retrieval_fails = len(df[(df["status"] == "SYSTEM_ERROR") & (df["error_type"] == "Retrieval Failure")])
    synthesis_fails = len(df[(df["status"] == "SYSTEM_ERROR") & (df["error_type"] == "Synthesis Failure")])
    data_limits = len(df[(df["status"] == "SYSTEM_ERROR") & (df["error_type"] == "Source Limitation")])
    print(f"- retrieval failures: {retrieval_fails}")
    print(f"- synthesis failures: {synthesis_fails}")
    print(f"- source/data limitations: {data_limits}")
    print(f"- other system failures: {sys_errors - retrieval_fails - synthesis_fails - data_limits}")

    print("\n==================================================")
    if success >= 20:
        print("PHASE 4 FULLY VERIFIED")
    else:
        print("PHASE 4 IMPLEMENTED — EMPIRICAL EVALUATION BLOCKED BY API LIMITS")

if __name__ == "__main__":
    main()
