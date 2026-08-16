"""
Grounded Answer Generation Module for GST Grounded RAG Assistant.

Uses Google Gemini (gemini-1.5-flash) to generate legally grounded,
citation-backed answers strictly constrained to retrieved evidence chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from config.settings import GEMINI_API_KEY, LLM_MODEL
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Result Data Structures
# ==============================================================================

@dataclass
class SourceCitation:
    """
    Metadata representation of an evidence source chunk.
    """
    chunk_id: str
    source_filename: str
    section_path: str
    page_start: int
    page_end: int


@dataclass
class GroundedAnswer:
    """
    Structured answer produced by GeminiGenerator with explicit source attribution.
    """
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    used_chunk_ids: List[str] = field(default_factory=list)
    confidence_label: str = "grounded"  # "grounded" | "insufficient_evidence"


# ==============================================================================
# 2. System Instructions & Prompt Construction
# ==============================================================================

SYSTEM_INSTRUCTION = """You are a senior GST legal assistant providing precise, authoritative answers on Indian Goods and Services Tax (GST) Circulars.

STRICT GROUNDING RULES:
1. Answer the user's question using ONLY the provided document evidence below.
2. Do NOT invent, assume, or extrapolate facts not directly stated in the evidence.
3. Do NOT use outside knowledge, general training memory, or unstated legal precedents.
4. If the provided evidence is insufficient, irrelevant, or does not contain the answer, explicitly state:
   "The answer cannot be determined from the provided documents."
5. Cite your evidence naturally where relevant using format: [{source_filename}, section {section_path}, page {page_start}].
6. Maintain technical accuracy, conciseness, and neutral tone. Do not provide speculative opinions."""


def format_evidence_context(retrieval_results: Sequence[Any]) -> Tuple[str, List[SourceCitation]]:
    """
    Format retrieval candidate results into structured evidence text and extract
    deduplicated SourceCitation objects.

    Args:
        retrieval_results: Sequence of SearchResult, HybridSearchResult, or Chunk objects.

    Returns:
        Tuple of (formatted_evidence_text, deduplicated_sources_list).
    """
    formatted_blocks: List[str] = []
    sources: List[SourceCitation] = []
    seen_chunk_ids: set[str] = set()

    for idx, item in enumerate(retrieval_results, start=1):
        chunk = item.chunk if hasattr(item, "chunk") else item
        if not hasattr(chunk, "chunk_id"):
            continue

        cid = chunk.chunk_id
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            sources.append(
                SourceCitation(
                    chunk_id=cid,
                    source_filename=chunk.source_filename,
                    section_path=chunk.section_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

        page_str = (
            f"page {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.page_start}-{chunk.page_end}"
        )
        
        block = (
            f"--- Document Evidence [{idx}] ---\n"
            f"Source File: {chunk.source_filename}\n"
            f"Section Path: {chunk.section_path} ({chunk.section_title})\n"
            f"Pages: {page_str}\n"
            f"Content:\n{chunk.text}\n"
        )
        formatted_blocks.append(block)

    evidence_text = "\n".join(formatted_blocks)
    return evidence_text, sources


def is_refusal_or_insufficient(answer_text: str) -> bool:
    """
    Check if the model response expresses inability to answer from the evidence.
    """
    if not answer_text or not answer_text.strip():
        return True

    text_lower = answer_text.lower()
    refusal_patterns = [
        "cannot be determined from the provided",
        "cannot be determined based on the provided",
        "not provided in the documents",
        "not mentioned in the provided",
        "provided documents do not contain",
        "provided evidence does not contain",
        "no information is provided",
        "insufficient evidence",
        "insufficient information",
        "not covered in the provided",
        "not found in the provided",
    ]
    return any(p in text_lower for p in refusal_patterns)


# ==============================================================================
# 3. Gemini Generator Class
# ==============================================================================

class GeminiGenerator:
    """
    Grounded answer generator orchestrating Gemini API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = LLM_MODEL,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize GeminiGenerator.

        Args:
            api_key: Google Gemini API key (defaults to config.settings.GEMINI_API_KEY).
            model_name: Configured model identifier (defaults to config.settings.LLM_MODEL).
            client: Optional pre-initialized Google GenAI client instance (useful for unit testing/mocking).
        """
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY
        self._client = client

    def _get_client(self) -> Any:
        """Initialize or return the cached Google GenAI client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set the GEMINI_API_KEY environment variable in .env or pass api_key."
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            raise RuntimeError(f"Failed to initialize Google GenAI client: {e}") from e

    def generate(
        self,
        query: str,
        retrieval_results: Sequence[Any],
        temperature: float = 0.0,
    ) -> GroundedAnswer:
        """
        Generate a grounded answer for the user query given retrieved evidence chunks.

        Workflow:
          1. Validates input query and retrieval candidates.
          2. Formats evidence context and extracts deduplicated source citations.
          3. Calls Gemini exactly once with strict system instructions.
          4. Returns structured GroundedAnswer with confidence categorization.

        Args:
            query: User question string.
            retrieval_results: Sequence of candidate chunks or SearchResults.
            temperature: LLM temperature (default: 0.0 for deterministic factual answers).

        Returns:
            GroundedAnswer containing the response, source citations, used chunk IDs, and confidence label.

        Raises:
            RuntimeError: If Gemini API fails during execution.
        """
        # 1. Input Validation
        if not query or not isinstance(query, str) or not query.strip():
            return GroundedAnswer(
                answer="No query provided.",
                sources=[],
                used_chunk_ids=[],
                confidence_label="insufficient_evidence",
            )

        if not retrieval_results:
            return GroundedAnswer(
                answer="The answer cannot be determined from the provided documents because no relevant evidence was found.",
                sources=[],
                used_chunk_ids=[],
                confidence_label="insufficient_evidence",
            )

        # 2. Format evidence and extract source citations
        evidence_text, sources = format_evidence_context(retrieval_results)
        used_ids = [s.chunk_id for s in sources]

        # 3. Construct prompt
        prompt = (
            f"Context Evidence:\n{evidence_text}\n\n"
            f"Question:\n{query.strip()}\n\n"
            f"Answer:"
        )

        # 4. Call Gemini API
        client = self._get_client()

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )

            # Extract response text
            answer_text = response.text if response and hasattr(response, "text") and response.text else ""
            answer_text = answer_text.strip()

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise RuntimeError(f"Gemini answer generation failed: {e}") from e

        # 5. Determine confidence label based on refusal/evidence analysis
        if not answer_text or is_refusal_or_insufficient(answer_text):
            confidence_label = "insufficient_evidence"
        else:
            confidence_label = "grounded"

        return GroundedAnswer(
            answer=answer_text,
            sources=sources,
            used_chunk_ids=used_ids,
            confidence_label=confidence_label,
        )
