import os
import logging
import requests
from typing import List, Dict, Any
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

def generate_answer(query: str, retrieved_chunks: List[RetrievalResult]) -> Dict[str, Any]:
    """
    Synthesize an answer using the Groq API, grounded entirely in the retrieved chunks.
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
        
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    
    if not retrieved_chunks:
        return {
            "answer": "I could not find an answer in the provided documents.",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    # Construct grounded context
    context_parts = []
    for chunk in retrieved_chunks:
        doc_id = getattr(chunk, 'document_id', None)
        if not doc_id:
            doc_id = chunk.source_file.split('/')[-1]
        page = chunk.page_number or 'Unknown'
        section = chunk.parent_id or 'Unknown'
        text = chunk.text
        context_parts.append(f"--- [Document: {doc_id}, Page: {page}, Section: {section}] ---\n{text}")
        
    context_str = "\n\n".join(context_parts)
    
    system_prompt = (
        "You are a strict financial analyst AI. Answer the user's question using ONLY the provided context.\n"
        "1. Do not invent financial figures or facts.\n"
        "2. Explicitly state if the context is insufficient to answer the question.\n"
        "3. Cite the relevant Document and Page/Section metadata in your answer.\n"
        "4. Prefer exact values from source tables when available.\n"
        "5. IMPORTANT: Identify discrepancies between documents (e.g. restated figures, different years). Do not silently reconcile conflicting numbers. Surface the conflict and explain which document provided each number."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
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
        "temperature": 0.0
    }
    
    logger.info(f"Sending generation request to Groq using model {model}")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=15
    )
    
    if response.status_code != 200:
        logger.error(f"Groq API Error: {response.status_code} - {response.text}")
        raise RuntimeError(f"LLM Generation failed with status code {response.status_code}")
        
    data = response.json()
    
    answer_text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    
    return {
        "answer": answer_text,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0)
    }
