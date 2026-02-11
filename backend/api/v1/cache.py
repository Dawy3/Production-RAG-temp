"""Semantic cache management endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.core.caching.semantic_cache import SemanticCache
from api.v1.dependencies import get_semantic_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cache"])


@router.delete("")
async def clear_semantic_cache(
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
):
    """Delete all semantic cache entries (RAM + Redis)."""
    if semantic_cache is None:
        raise HTTPException(status_code=404, detail="Semantic cache is not enabled")

    stats_before = semantic_cache.get_stats()
    entries_cleared = stats_before["cache_size"]

    semantic_cache.clear()

    logger.info("Semantic cache cleared: %d entries removed", entries_cleared)

    return {
        "message": "Semantic cache cleared",
        "entries_cleared": entries_cleared,
    }
