"""
Multi-Query Transformation for RAG Pipeline.

Generate 3-5 query variations for ambiguous queries to improve recall.
Expected improvement: +15-25% recall with multi-query approach.

Usage:
    transformer = MultiQueryTransformer()

    # For ambiguous queries, generate variations
    if is_ambiguous:
        result = await transformer.transform(query)
        queries = result.all_queries  # Original + variations
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class TransformedQuery:
    """Result of multi-query transformation."""

    original_query: str
    variations: list[str]

    @property
    def all_queries(self) -> list[str]:
        """Get all queries including original (deduplicated)."""
        queries = [self.original_query] + self.variations
        seen = set()
        unique = []
        for q in queries:
            q_normalized = q.lower().strip()
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique.append(q)
        return unique

    @property
    def count(self) -> int:
        """Total number of unique queries."""
        return len(self.all_queries)


MULTI_QUERY_PROMPT = """Generate {num_variations} different versions of this search query to improve document retrieval.
Each variation should:
- Capture the same intent but use different words/phrasing
- Include synonyms or related terms
- Vary between question forms and keyword-based queries

Original query: {query}

Return ONLY the variations, one per line, no numbering or bullets."""


class MultiQueryTransformer:
    """
    Generate 3-5 query variations for ambiguous queries.

    Use this when:
    - Query is ambiguous or could be interpreted multiple ways
    - Initial retrieval returned poor results
    - User query is very short or vague

    Example:
        transformer = MultiQueryTransformer()
        result = await transformer.transform("python error")
        # result.all_queries might be:
        # ["python error", "python exception handling", "fix python bug",
        #  "python traceback solution", "debug python code"]
    """

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4o-mini",
        num_variations: int = 4,
    ):
        """
        Args:
            client: OpenAI client (creates one if not provided)
            model: Model for generating variations (use cheap/fast model)
            num_variations: Number of variations to generate (3-5 recommended)
        """
        self.client = client or AsyncOpenAI()
        self.model = model
        self.num_variations = min(max(num_variations, 3), 5)  # Clamp to 3-5

    async def transform(self, query: str) -> TransformedQuery:
        """
        Generate query variations.

        Args:
            query: Original search query

        Returns:
            TransformedQuery with original + variations
        """
        query = query.strip()
        if not query:
            return TransformedQuery(original_query="", variations=[])

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": MULTI_QUERY_PROMPT.format(
                            query=query,
                            num_variations=self.num_variations,
                        ),
                    }
                ],
                temperature=0.7,  # Some creativity for diverse variations
                max_tokens=200,
            )

            content = response.choices[0].message.content or ""

            # Parse variations (one per line)
            variations = [
                line.strip().lstrip("0123456789.-) ")
                for line in content.split("\n")
                if line.strip() and line.strip().lower() != query.lower()
            ]

            # Limit to requested number
            variations = variations[:self.num_variations]

            return TransformedQuery(
                original_query=query,
                variations=variations,
            )

        except Exception as e:
            logger.warning(f"Multi-query transformation failed: {e}")
            # Return original only on failure
            return TransformedQuery(original_query=query, variations=[])


def create_transformer(
    model: str = "gpt-4o-mini",
    num_variations: int = 4,
) -> MultiQueryTransformer:
    """
    Factory function to create multi-query transformer.

    Args:
        model: OpenAI model for generating variations
        num_variations: Number of variations (3-5)

    Example:
        transformer = create_transformer()

        # Use with ambiguous queries
        if classification.route == QueryRoute.CLARIFICATION:
            result = await transformer.transform(query)
            # Search with all variations
            for q in result.all_queries:
                results.extend(await retriever.search(q))
    """
    return MultiQueryTransformer(model=model, num_variations=num_variations)
