import time
import logging
from src.retrieval.dense import DenseRetriever
from src.generation.synthesize import generate_answer
from src.retrieval.answer import AnswerResult

logger = logging.getLogger(__name__)

class BaselineSystem:
    """
    BASELINE SYSTEM
    - Dense-only retrieval (Qdrant) using bge-small-en-v1.5
    - NO BM25
    - NO RRF
    - NO Cross-Encoder Reranking
    - NO Semantic Cache
    - Top-5 chunks fed to LLM synthesis.
    """
    def __init__(self):
        self.retriever = DenseRetriever()
        
    def answer(self, query: str) -> AnswerResult:
        logger.info(f"[Baseline] Answering query: '{query}'")
        start_total = time.time()
        
        # 1. Dense Retrieval (Top 5)
        start_retrieval = time.time()
        context = self.retriever.retrieve(query, top_k=5)
        retrieval_latency = (time.time() - start_retrieval) * 1000
        
        # 2. Answer Generation
        start_gen = time.time()
        try:
            gen_result = generate_answer(query, context)
            generation_latency = (time.time() - start_gen) * 1000
            
            return AnswerResult(
                answer=gen_result["answer"],
                cache_hit=False,
                cache_distance=None,
                cache_latency=0.0,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                prompt_tokens=gen_result["prompt_tokens"],
                completion_tokens=gen_result["completion_tokens"],
                total_tokens=gen_result["total_tokens"],
                context=context
            )
            
        except Exception as e:
            logger.error(f"[Baseline] Generation failed: {e}")
            generation_latency = (time.time() - start_gen) * 1000
            return AnswerResult(
                answer=f"Generation failed: {e}",
                cache_hit=False,
                cache_distance=None,
                cache_latency=0.0,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=(time.time() - start_total) * 1000,
                context=context
            )

if __name__ == "__main__":
    baseline = BaselineSystem()
    res = baseline.answer("What were Apple's total net sales in the Americas segment for 2025?")
    print(res.answer)
