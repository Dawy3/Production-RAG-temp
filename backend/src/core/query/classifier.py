"""
مصنف الاستعلامات للمشروع العربي - Rule-Based Query Classification for Arabic RAG Pipeline.

تصنيف الاستعلامات:
- RETRIEVAL: أسئلة واقعية يمكن الإجابة عليها من المستندات ← استخدام RAG
- GENERATION: مهام إبداعية، تلخيص، تحليل، أو تحيات ← LLM فقط
- CLARIFICATION: غامض ← طلب توضيح
- REJECTION: خارج النطاق ← رفض بلطف

Cost: $0 | Latency: <1ms
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    """تصنيفات توجيه الاستعلامات."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CLARIFICATION = "clarification"
    REJECTION = "rejection"


@dataclass
class ClassificationResult:
    """نتيجة تصنيف الاستعلام."""
    query: str
    route: QueryRoute
    reason: str = ""
    follow_up_question: Optional[str] = None

    @property
    def needs_rag(self) -> bool:
        return self.route == QueryRoute.RETRIEVAL

    @property
    def needs_clarification(self) -> bool:
        return self.route == QueryRoute.CLARIFICATION


class QueryClassifier:
    """
    مصنف استعلامات عربي قائم على القواعد.

    تصنيف سريع باستخدام الأنماط - بدون استدعاء LLM.
    Cost: $0 | Latency: <1ms
    """

    # أنماط التحيات والترحيب - يتم تمريرها إلى LLM للرد بترحيب
    GREETING_PATTERNS = [
        r"^(مرحبا|مرحبا بك|مرحباً)$",
        r"^(اهلا|أهلا|اهلاً|أهلاً).*$",
        r"^(السلام عليكم|السلام).*$",
        r"^(صباح الخير|مساء الخير|مساء النور|صباح النور)$",
        r"^(هلا|هلا والله|يا هلا|حياك|حياك الله)$",
        r"^(كيف حالك|كيف الحال|شلونك|شخبارك).*$",
        r"^(hi|hello|hey|good morning|good evening)$",
    ]

    # أنماط التوليد (مهام إبداعية - لا تحتاج RAG)
    GENERATION_PATTERNS = [
        r"^(اكتب|أنشئ|ولد|صمم|حرر)\s",
        r"^(لخص|اختصر)\s",
        r"^(ترجم|حول)\s",
        r"^(اشرح|وصف)\s.*(لي|ببساطة|كأن)",
        r"(بكلماتك|بأسلوبك|ببساطة)$",
        r"^(ما رأيك|أعطني رأيك)",
    ]

    # كلمات مرفوضة (خارج النطاق)
    REJECTION_KEYWORDS = [
        # ضار
        "اختراق", "تهكير", "هاك", "فيروس", "سرقة", "تجسس",
        "hack", "crack", "exploit", "malware", "virus",
        # خارج الموضوع
        "طقس", "أسهم", "رياضة", "مشاهير", "وصفة طبخ", "نكتة", "لعبة",
    ]

    # استعلامات غامضة تحتاج توضيح
    VAGUE_QUERIES = [
        "مساعدة", "ساعدني", "أحتاج مساعدة",
        "سؤال", "عندي سؤال",
        "نعم", "لا", "أوكي", "حسنا", "طيب", "تمام",
        "ماذا", "كيف", "لماذا", "ايش", "وش", "شنو",
    ]

    MIN_QUERY_WORDS = 2

    def __init__(
        self,
        rejection_keywords: Optional[list[str]] = None,
        generation_patterns: Optional[list[str]] = None,
        min_query_words: int = 2,
    ):
        self.rejection_keywords = self.REJECTION_KEYWORDS.copy()
        if rejection_keywords:
            self.rejection_keywords.extend(rejection_keywords)

        self.greeting_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS
        ]

        self.generation_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GENERATION_PATTERNS
        ]
        if generation_patterns:
            self.generation_patterns.extend([
                re.compile(p, re.IGNORECASE) for p in generation_patterns
            ])

        self.min_query_words = min_query_words

    def _is_greeting(self, query: str) -> bool:
        """تحقق مما إذا كان الاستعلام تحية أو ترحيب."""
        for pattern in self.greeting_patterns:
            if pattern.search(query):
                return True
        return False

    def classify(self, query: str) -> ClassificationResult:
        """
        تصنيف الاستعلام باستخدام القواعد.

        Args:
            query: استعلام المستخدم

        Returns:
            ClassificationResult مع المسار
        """
        query = query.strip()
        query_lower = query.lower()

        # استعلام فارغ
        if not query:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="استعلام فارغ",
                follow_up_question="مرحباً! كيف يمكنني مساعدتك اليوم؟",
            )

        # التحيات والترحيب → يتم تمريرها إلى LLM للرد بترحيب
        if self._is_greeting(query):
            return ClassificationResult(
                query=query,
                route=QueryRoute.GENERATION,
                reason="تحية أو ترحيب",
            )

        # التحقق من الرفض أولاً (ضار/خارج النطاق)
        for keyword in self.rejection_keywords:
            if keyword in query_lower:
                return ClassificationResult(
                    query=query,
                    route=QueryRoute.REJECTION,
                    reason=f"يحتوي على كلمة خارج النطاق: {keyword}",
                )

        # التحقق من التوضيح (غامض جداً)
        words = query.split()
        if len(words) < self.min_query_words:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="استعلام قصير جداً",
                follow_up_question="هل يمكنك تقديم مزيد من التفاصيل حول ما تبحث عنه؟",
            )

        if query in self.VAGUE_QUERIES:
            return ClassificationResult(
                query=query,
                route=QueryRoute.CLARIFICATION,
                reason="استعلام غامض",
                follow_up_question="هل يمكنك أن تكون أكثر تحديداً حول ما تحتاج المساعدة فيه؟",
            )

        # التحقق من التوليد (مهام إبداعية - لا تحتاج RAG)
        for pattern in self.generation_patterns:
            if pattern.search(query):
                return ClassificationResult(
                    query=query,
                    route=QueryRoute.GENERATION,
                    reason="مهمة إبداعية/توليدية",
                )

        # الافتراضي: استرجاع (استخدام RAG)
        return ClassificationResult(
            query=query,
            route=QueryRoute.RETRIEVAL,
            reason="استعلام واقعي - استخدام RAG",
        )


def create_classifier(
    rejection_keywords: Optional[list[str]] = None,
    min_query_words: int = 2,
) -> QueryClassifier:
    """
    دالة إنشاء مصنف الاستعلامات.

    Args:
        rejection_keywords: كلمات إضافية للرفض
        min_query_words: الحد الأدنى لكلمات الاستعلام الصالح
    """
    return QueryClassifier(
        rejection_keywords=rejection_keywords,
        min_query_words=min_query_words,
    )
