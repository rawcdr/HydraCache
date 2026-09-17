import os
import json
import uuid
import time
import logging
from typing import Optional, Dict, Any
from redis import Redis
from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError
from fastembed.embedding import TextEmbedding

logger = logging.getLogger(__name__)

class CacheResult:
    def __init__(self, hit: bool, answer: Optional[str] = None, 
                 distance: Optional[float] = None, cache_id: Optional[str] = None, 
                 latency: float = 0.0):
        self.hit = hit
        self.answer = answer
        self.distance = distance
        self.cache_id = cache_id
        self.latency = latency

class SemanticCache:
    def __init__(self, redis_url: str = None, threshold: float = 0.15, ttl: int = 86400):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.threshold = threshold
        self.ttl = ttl
        self.index_name = "idx:semantic_cache"
        self.key_prefix = "cache:"
        self.vector_dim = 384
        self.distance_metric = "COSINE"
        
        # Initialize embedding model (same as retrieval)
        try:
            self.embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            logger.info("Initialized cache embedding model: BAAI/bge-small-en-v1.5")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
        try:
            self.redis = Redis.from_url(self.redis_url, decode_responses=False)
            self._create_index()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.redis = None

    def _create_index(self):
        """Idempotent creation of the Redis Stack vector index."""
        schema = (
            TextField("query_text"),
            TextField("answer"),
            NumericField("timestamp"),
            VectorField(
                "query_embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": self.distance_metric
                }
            )
        )
        
        try:
            self.redis.ft(self.index_name).info()
            logger.info(f"Index {self.index_name} already exists.")
        except ResponseError as e:
            if "unknown index name" in str(e).lower():
                logger.info(f"Creating index {self.index_name}...")
                self.redis.ft(self.index_name).create_index(
                    schema,
                    definition=IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)
                )
            else:
                logger.error(f"Error checking index: {e}")
                raise

    def lookup(self, query: str) -> CacheResult:
        """Search the cache for semantically similar queries."""
        start_time = time.time()
        
        if not self.redis:
            logger.warning("Redis unavailable, bypassing cache lookup.")
            return CacheResult(hit=False, latency=(time.time() - start_time)*1000)
            
        try:
            # 1. Generate query embedding
            query_embedding = list(self.embedding_model.embed([query]))[0]
            embedding_bytes = query_embedding.astype("float32").tobytes()
            
            # 2. Search Redis Vector Index
            from redis.commands.search.query import Query
            search_query = (
                Query("*=>[KNN 1 @query_embedding $vec AS distance]")
                .return_fields("distance", "answer")
                .sort_by("distance")
                .dialect(2)
            )
            
            res = self.redis.ft(self.index_name).search(
                search_query, 
                query_params={"vec": embedding_bytes}
            )
            
            if len(res.docs) > 0:
                doc = res.docs[0]
                distance = float(doc.distance)
                
                # Compare against threshold
                if distance < self.threshold:
                    logger.info(f"Cache HIT for query '{query}' (distance: {distance:.4f})")
                    return CacheResult(
                        hit=True,
                        answer=doc.answer.decode("utf-8") if isinstance(doc.answer, bytes) else doc.answer,
                        distance=distance,
                        cache_id=doc.id,
                        latency=(time.time() - start_time)*1000
                    )
                else:
                    logger.info(f"Cache MISS for query '{query}' (best distance: {distance:.4f} > {self.threshold})")
                    return CacheResult(
                        hit=False,
                        distance=distance,
                        latency=(time.time() - start_time)*1000
                    )
            
            logger.info(f"Cache MISS for query '{query}' (empty cache)")
            return CacheResult(hit=False, latency=(time.time() - start_time)*1000)
            
        except Exception as e:
            logger.error(f"Semantic lookup failed: {e}")
            return CacheResult(hit=False, latency=(time.time() - start_time)*1000)

    def store(self, query: str, answer: str) -> bool:
        """Store a successful query and answer in the cache."""
        if not self.redis:
            logger.warning("Redis unavailable, bypassing cache store.")
            return False
            
        if not answer or not answer.strip():
            logger.warning("Attempted to cache an empty answer. Skipping.")
            return False
            
        try:
            query_embedding = list(self.embedding_model.embed([query]))[0]
            embedding_bytes = query_embedding.astype("float32").tobytes()
            
            cache_id = f"{self.key_prefix}{uuid.uuid4()}"
            
            mapping = {
                "query_text": query,
                "answer": answer,
                "timestamp": int(time.time()),
                "query_embedding": embedding_bytes
            }
            
            # Store hash in Redis
            self.redis.hset(cache_id, mapping=mapping)
            # Set TTL
            self.redis.expire(cache_id, self.ttl)
            
            logger.info(f"Cached answer for query '{query}' with TTL {self.ttl}s")
            return True
        except Exception as e:
            logger.error(f"Failed to store cache record: {e}")
            return False
            
    def clear(self):
        """Clear all semantic cache entries."""
        if not self.redis:
            return
        
        try:
            # We can find all keys matching the prefix and delete them
            keys = self.redis.keys(f"{self.key_prefix}*")
            if keys:
                self.redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} entries from semantic cache.")
        except Exception as e:
            logger.error(f"Failed to clear semantic cache: {e}")
