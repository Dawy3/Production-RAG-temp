"""
Semantic Cache for RAG Pipeline.

3-Layer Architecture:
- Layer 1: Exact match (hash lookup - fast)
- Layer 2: Semantic similarity (embedding search, threshold > 0.9)
- Layer 3: Cross-encoder validation (filter false positives)

Redis is the default persistent store. Falls back to RAM if Redis is unavailable.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis
from sentence_transformers import CrossEncoder

from src.config import settings
from src.core.embedding.generator import create_embedding_generator

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached query-response pair."""
    query: str
    response: str
    embedding: list[float]
    metadata: dict
    created_at: float


@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    response: Optional[str] = None
    layer: Optional[str] = None  # "exact", "semantic", None
    similarity: float = 0.0
    latency_ms: float = 0.0


class SemanticCache:
    """
    Self-contained 3-layer semantic cache.

    Reads all config from settings. Redis is default, RAM is fallback.

    Usage:
        cache = SemanticCache()

        result = await cache.get(query)
        if result.hit:
            return result.response

        response = await generate(query)
        await cache.set(query, response)
    """

    def __init__(self):
        # Config
        self.similarity_threshold = settings.cache.semantic_cache_threshold
        self.rerank_threshold = settings.cache.rerank_threshold
        self.ttl = settings.cache.semantic_cache_ttl
        self.max_size = settings.cache.semantic_cache_max_size
        self.prefix = "sem_cache"

        # Embedding function
        self._embedding_generator = create_embedding_generator()
        self._embed_func = self._embedding_generator.embed_query

        # Redis (default) with RAM fallback
        self.redis = None
        self._using_redis = False
        if settings.cache.redis_url:
            try:
                self.redis = redis.from_url(settings.cache.redis_url, decode_responses=True)
                self.redis.ping()
                self._using_redis = True
                logger.info("Semantic cache: Redis connected (%s)", settings.cache.redis_url)
            except Exception as e:
                logger.warning("Semantic cache: Redis unavailable (%s) — using RAM", e)
                self.redis = None
        else:
            logger.info("Semantic cache: No REDIS_URL configured — using RAM")

        # In-memory index (always needed for semantic similarity search)
        self._memory_cache: dict[str, CacheEntry] = {}
        self._embeddings: list[tuple[str, np.ndarray]] = []

        # Layer 3: Cross-encoder
        try:
            model_name = settings.cache.cross_encoder_model
            self.reranker = CrossEncoder(model_name)
            logger.info("Loaded cross-encoder: %s", model_name)
        except Exception as e:
            logger.warning("Cross-encoder failed to load (%s) — Layer 3 disabled", e)
            self.reranker = None

        # Stats
        self._stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0}

        # Warm up memory index from Redis
        if self._using_redis:
            self._warm_from_redis()

        logger.info(
            "SemanticCache ready: storage=%s, threshold=%.2f, reranker=%s",
            "redis" if self._using_redis else "ram",
            self.similarity_threshold,
            "yes" if self.reranker else "no",
        )

    async def get(self, query: str) -> CacheResult:
        """Look up query in cache (3 layers)."""
        start = time.perf_counter()

        # Layer 1: Exact match
        exact_result = self._exact_match(query)
        if exact_result:
            self._stats["hits_exact"] += 1
            return CacheResult(
                hit=True,
                response=exact_result,
                layer="exact",
                similarity=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Layer 2: Semantic similarity
        query_embedding = np.array(await self._embed_func(query))
        semantic_result = self._semantic_match(query, query_embedding)

        if semantic_result:
            self._stats["hits_semantic"] += 1
            return CacheResult(
                hit=True,
                response=semantic_result["response"],
                layer="semantic",
                similarity=semantic_result["similarity"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        self._stats["misses"] += 1
        return CacheResult(hit=False, latency_ms=(time.perf_counter() - start) * 1000)

    async def set(self, query: str, response: str, metadata: Optional[dict] = None) -> None:
        """Cache a query-response pair."""
        if not response or not response.strip():
            return

        query_hash = self._hash_query(query)

        try:
            embedding = await self._embed_func(query)
        except Exception as e:
            logger.error("Failed to embed query for caching: %s", e)
            return

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata or {},
            created_at=time.time(),
        )

        # Store in Redis + memory, or just memory
        if self._using_redis:
            try:
                self._redis_set(query_hash, entry)
            except Exception as e:
                logger.error("Redis write failed (%s) — storing in RAM only", e)
                self._memory_set(query_hash, entry)
        else:
            self._memory_set(query_hash, entry)

    # --- Layer 1: Exact match ---

    def _exact_match(self, query: str) -> Optional[str]:
        """Hash-based exact match."""
        query_hash = self._hash_query(query)

        # Check Redis first
        if self._using_redis:
            try:
                data = self.redis.get(f"{self.prefix}:exact:{query_hash}")
                if data:
                    entry = json.loads(data)
                    response = entry.get("response", "")
                    if response and response.strip():
                        return response
                    self.redis.delete(f"{self.prefix}:exact:{query_hash}")
            except Exception:
                pass

        # Fallback to memory
        if query_hash in self._memory_cache:
            response = self._memory_cache[query_hash].response
            if response and response.strip():
                return response
            del self._memory_cache[query_hash]
            self._embeddings = [(k, e) for k, e in self._embeddings if k != query_hash]

        return None

    # --- Layer 2 + 3: Semantic match ---

    def _semantic_match(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        """Semantic similarity with cross-encoder validation."""
        candidates = self._find_similar(query_embedding)
        if not candidates:
            return None

        best = candidates[0]
        if best["similarity"] < self.similarity_threshold:
            return None

        # Layer 3: Cross-encoder validation
        if self.reranker:
            try:
                score = self.reranker.predict([(query, best["query"])])[0]
                if score < self.rerank_threshold:
                    return None
            except Exception:
                pass  # Fail open

        return best

    def _find_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Find similar queries using cosine similarity."""
        if not self._embeddings:
            return []

        similarities = []
        for key, emb in self._embeddings:
            norm_a = np.linalg.norm(query_embedding)
            norm_b = np.linalg.norm(emb)
            if norm_a == 0 or norm_b == 0:
                continue
            sim = float(np.dot(query_embedding, emb) / (norm_a * norm_b))
            similarities.append((key, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for key, sim in similarities[:top_k]:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry.response and entry.response.strip():
                    results.append({"query": entry.query, "response": entry.response, "similarity": sim})
        return results

    # --- Storage ---

    def _hash_query(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _memory_set(self, key: str, entry: CacheEntry) -> None:
        if len(self._memory_cache) >= self.max_size:
            self._evict_oldest()
        self._memory_cache[key] = entry
        self._embeddings.append((key, np.array(entry.embedding)))

    def _redis_set(self, key: str, entry: CacheEntry) -> None:
        data = {
            "query": entry.query,
            "response": entry.response,
            "embedding": entry.embedding,
            "metadata": entry.metadata,
            "created_at": entry.created_at,
        }
        self.redis.setex(f"{self.prefix}:exact:{key}", self.ttl, json.dumps(data))
        # Also keep in memory for semantic search
        self._memory_set(key, entry)

    def _evict_oldest(self) -> None:
        if not self._memory_cache:
            return
        oldest_key = min(self._memory_cache, key=lambda k: self._memory_cache[k].created_at)
        del self._memory_cache[oldest_key]
        self._embeddings = [(k, e) for k, e in self._embeddings if k != oldest_key]

    def _warm_from_redis(self) -> None:
        """Load cached entries from Redis into memory for semantic search."""
        try:
            keys = self.redis.keys(f"{self.prefix}:exact:*")
            if not keys:
                return

            loaded = 0
            for key in keys:
                try:
                    data = self.redis.get(key)
                    if not data:
                        continue
                    entry_data = json.loads(data)
                    response = entry_data.get("response", "")
                    embedding = entry_data.get("embedding")
                    if not response or not response.strip() or not embedding:
                        continue

                    query_hash = key.split(":")[-1]
                    entry = CacheEntry(
                        query=entry_data["query"],
                        response=response,
                        embedding=embedding,
                        metadata=entry_data.get("metadata", {}),
                        created_at=entry_data.get("created_at", time.time()),
                    )
                    self._memory_cache[query_hash] = entry
                    self._embeddings.append((query_hash, np.array(embedding)))
                    loaded += 1
                except Exception as e:
                    logger.warning("Failed to load cache entry %s: %s", key, e)

            logger.info("Redis warm-up: loaded %d entries", loaded)
        except Exception as e:
            logger.warning("Redis warm-up failed: %s", e)

    # --- Stats ---

    def get_stats(self) -> dict:
        total = sum(self._stats.values())
        hit_rate = (self._stats["hits_exact"] + self._stats["hits_semantic"]) / total if total > 0 else 0
        return {
            "hits_exact": self._stats["hits_exact"],
            "hits_semantic": self._stats["hits_semantic"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self._memory_cache),
            "storage": "redis" if self._using_redis else "ram",
        }

    def clear(self) -> None:
        self._memory_cache.clear()
        self._embeddings.clear()
        self._stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0}
        if self._using_redis:
            try:
                keys = self.redis.keys(f"{self.prefix}:*")
                if keys:
                    self.redis.delete(*keys)
            except Exception:
                pass
