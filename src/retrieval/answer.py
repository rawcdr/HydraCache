import time
import logging
from typing import List, Optional
from dataclasses import dataclass
from src.retrieval.models import RetrievalResult
from src.retrieval.pipeline import PipelineRetriever
from src.cache.semantic_cache import SemanticCache
from src.generation.synthesize import generate_answer

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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context: Optional[List[RetrievalResult]] = None


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
        try:
            gen_result = generate_answer(query, context)
            generation_latency = (time.time() - start_gen) * 1000
            
            # 4. Cache the successful result
            self.cache.store(query, gen_result["answer"])
            
            return AnswerResult(
                answer=gen_result["answer"],
                cache_hit=False,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                prompt_tokens=gen_result["prompt_tokens"],
                completion_tokens=gen_result["completion_tokens"],
                total_tokens=gen_result["total_tokens"],
                context=context
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            generation_latency = (time.time() - start_gen) * 1000
            # Do NOT cache failed requests
            return AnswerResult(
                answer=f"Generation failed: {e}",
                cache_hit=False,
                cache_distance=cache_res.distance,
                cache_latency=cache_latency,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                context=context
            )
