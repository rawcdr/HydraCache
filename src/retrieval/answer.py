import time
import logging
from typing import List, Optional
from dataclasses import dataclass
from src.retrieval.models import RetrievalResult
from src.retrieval.pipeline import PipelineRetriever
from src.cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

@dataclass
class AnswerResult:
    answer: str
    cache_hit: bool
    cache_distance: Optional[float] = None
    cache_latency: float = 0.0
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    context: Optional[List[RetrievalResult]] = None

def generate_answer(query: str, retrieved_chunks: List[RetrievalResult]) -> str:
    """
    A lightweight deterministic stub for LLM answer generation.
    In a real implementation, this would call OpenAI/Anthropic etc.
    """
    if not retrieved_chunks:
        return "I could not find an answer in the provided documents."
        
    # Generate a dummy answer that clearly shows it used the chunks
    sources = []
    for chunk in retrieved_chunks:
        page = chunk.page_number or 'Unknown'
        sources.append(f"[Page {page}]")
        
    source_str = ", ".join(sources)
    answer = f"Based on the retrieved context {source_str}, this is a generated answer for: '{query}'."
    return answer

class AnswerPipeline:
    def __init__(self, pipeline: Optional[PipelineRetriever] = None, cache: Optional[SemanticCache] = None):
        self.pipeline = pipeline or PipelineRetriever()
        self.cache = cache or SemanticCache()
        
    def answer(self, query: str) -> AnswerResult:
        logger.info(f"Answering query: '{query}'")
        start_total = time.time()
        
        # 1. Semantic Cache Lookup
        cache_res = self.cache.lookup(query)
        cache_latency = cache_res.latency
        
        if cache_res.hit and cache_res.answer:
            logger.info("Cache hit successfully bypassed retrieval and generation.")
            return AnswerResult(
                answer=cache_res.answer,
                cache_hit=True,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=0.0,
                generation_latency=0.0,
                total_latency=(time.time() - start_total) * 1000,
                context=[] # Context was already baked into the answer
            )
            
        # 2. Phase 2 Retrieval (Cache Miss)
        start_retrieval = time.time()
        context = self.pipeline.retrieve(query)
        retrieval_latency = (time.time() - start_retrieval) * 1000
        
        # 3. Answer Generation
        start_gen = time.time()
        generated_text = generate_answer(query, context)
        generation_latency = (time.time() - start_gen) * 1000
        
        # 4. Cache the successful result
        self.cache.store(query, generated_text)
        
        return AnswerResult(
            answer=generated_text,
            cache_hit=False,
            cache_distance=cache_res.distance, # If it was close but not enough, or None
            cache_latency=cache_latency,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            total_latency=(time.time() - start_total) * 1000,
            context=context
        )
