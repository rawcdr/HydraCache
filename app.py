import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.retrieval.pipeline import PipelineRetriever
from src.cache.semantic_cache import SemanticCache
from src.retrieval.answer import AnswerPipeline

# Page Config
st.set_page_config(page_title="HydraCache Demo", layout="wide")
load_dotenv()

@st.cache_resource
def get_pipeline():
    pipeline = PipelineRetriever()
    cache = SemanticCache()
    return AnswerPipeline(pipeline=pipeline, cache=cache)

def render_search():
    st.header("Search Interface")
    
    query = st.text_input("Enter your query:", "What were Apple's net sales in 2025?")
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            pipeline = get_pipeline()
            res = pipeline.answer(query)
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Answer")
                st.write(res.answer)
                
                if res.context:
                    st.subheader("Retrieved Context")
                    for i, chunk in enumerate(res.context):
                        with st.expander(f"Chunk {i+1} - Page {chunk.page_number} (Score: {chunk.score:.2f})"):
                            st.write(chunk.text)
                            st.caption(f"Section: {chunk.parent_id}")
                            
            with col2:
                st.subheader("Telemetry")
                st.metric("Cache Status", "HIT" if res.cache_hit else "MISS")
                if res.cache_distance is not None:
                    st.metric("Cache Distance", f"{res.cache_distance:.4f}")
                    
                st.write("### Latency (ms)")
                st.write(f"- Cache Lookup: {res.cache_latency:.2f}")
                st.write(f"- Retrieval: {res.retrieval_latency:.2f}")
                st.write(f"- Generation: {res.generation_latency:.2f}")
                st.write(f"- **Total: {res.total_latency:.2f}**")
                
                st.write("### Token Usage")
                st.write(f"- Prompt: {res.prompt_tokens}")
                st.write(f"- Completion: {res.completion_tokens}")
                st.write(f"- Total: {res.total_tokens}")

def render_analytics():
    st.header("Evaluation & Benchmarking Analytics")
    
    results_path = "eval/results.json"
    if not os.path.exists(results_path):
        st.warning("No evaluation results found. Run `python eval/run_eval.py` first.")
        return
        
    with open(results_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            
    if len(data) < 40:
        st.warning(f"Evaluation in progress — {len(data)}/40 completed. Results not final.")
        if len(data) == 0:
            return
            
    df = pd.DataFrame(data)
    
    # Overview
    st.subheader("Baseline vs Optimized (HydraCache)")
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "latency", "total_tokens"]
    avg_df = df.groupby("system")[metrics].mean().reset_index()
    
    st.dataframe(avg_df.style.format({
        "faithfulness": "{:.2f}",
        "answer_relevancy": "{:.2f}",
        "context_precision": "{:.2f}",
        "context_recall": "{:.2f}",
        "latency": "{:.0f} ms",
        "total_tokens": "{:.0f}"
    }), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category Breakdown
        st.subheader("Metrics by Category (Optimized)")
        opt_df = df[df["system"] == "Optimized"]
        cat_df = opt_df.groupby("category")[["faithfulness", "answer_relevancy"]].mean().reset_index()
        fig1 = px.bar(cat_df, x="category", y=["faithfulness", "answer_relevancy"], barmode="group", title="Quality by Category")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Latency Distribution
        st.subheader("Latency Distribution")
        fig2 = px.box(df, x="system", y="latency", points="all", title="Total Latency (ms) by System")
        st.plotly_chart(fig2, use_container_width=True)

def render_failure_analysis():
    st.header("Failure Analysis")
    st.write("Review queries that failed or performed poorly in evaluation.")
    
    analysis_path = "eval/failure_analysis.json"
    if not os.path.exists(analysis_path):
        st.info("No failure analysis data found yet. Please complete the failure analysis step.")
        return
        
    with open(analysis_path, "r", encoding="utf-8") as f:
        failures = json.load(f)
        
    for fail in failures:
        with st.expander(f"Question: {fail['question']} ({fail['category']})"):
            st.write(f"**Classification:** {fail['classification']}")
            st.write(f"**Expected Answer:** {fail['expected_answer']}")
            st.write(f"**Actual Answer:** {fail['actual_answer']}")
            st.write(f"**Attempted Fix:** {fail.get('attempted_fix', 'None')}")
            
            st.subheader("Retrieved Context")
            for chunk in fail['retrieved_chunks']:
                st.caption(f"Chunk: {chunk[:200]}...")

def main():
    st.title("HydraCache Dashboard")
    
    tabs = st.tabs(["Search", "Analytics", "Failure Analysis"])
    
    with tabs[0]:
        render_search()
        
    with tabs[1]:
        render_analytics()
        
    with tabs[2]:
        render_failure_analysis()

if __name__ == "__main__":
    main()
