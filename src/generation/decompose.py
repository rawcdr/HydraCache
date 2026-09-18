import os
import json
import logging
import requests
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class QueryPlan(BaseModel):
    mode: Literal["single_hop", "multi_hop"] = Field(description="The classification of the query.")
    sub_questions: List[str] = Field(default_factory=list, description="List of decomposed sub-questions if multi-hop. Empty if single_hop.")

def analyze_query(query: str) -> QueryPlan:
    """
    Analyzes a query to determine if it requires multiple hops (e.g. crossing time periods or documents).
    If multi_hop, it decomposes the query into sub-questions.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
        
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    
    system_prompt = (
        "You are a query decomposition AI for a RAG system. Your goal is to analyze a user query and determine if it requires information from multiple documents or distinct concepts (multi_hop) or if it can be answered by a single semantic search (single_hop).\n"
        "Return ONLY a JSON object matching this schema:\n"
        "{\n"
        "  \"mode\": \"single_hop\" or \"multi_hop\",\n"
        "  \"sub_questions\": [\"sub-question 1\", \"sub-question 2\"] (only if multi_hop)\n"
        "}\n"
        "Rule 1: Use 'multi_hop' if the query compares figures across years (e.g., '2024 vs 2025') or asks for changes between time periods.\n"
        "Rule 2: Do NOT hallucinate unnecessary sub-questions. Only break it down into the core facts needed.\n"
        "Rule 3: Ensure the output is valid JSON."
    )
    
    user_prompt = f"Query: {query}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    logger.info(f"Analyzing query for multi-hop mode...")
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        parsed = json.loads(content)
        plan = QueryPlan(**parsed)
        logger.info(f"Query analysis complete: {plan.mode} with {len(plan.sub_questions)} sub-questions.")
        return plan
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}. Falling back to single_hop.")
        return QueryPlan(mode="single_hop", sub_questions=[])
