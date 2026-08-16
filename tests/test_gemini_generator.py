"""
Unit tests for GeminiGenerator in generation/gemini_generator.py.
Uses mock clients to avoid real API calls in the unit test suite.
Covers valid grounded responses, insufficient evidence refusals,
empty retrieval inputs, prompt formatting, source deduplication,
API error wrapping, malformed response handling, and chunk immutability.
"""

import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from ingestion.chunker import Chunk
from retrieval.bm25_retriever import SearchResult
from retrieval.hybrid_retriever import HybridSearchResult
from generation.gemini_generator import (
    GeminiGenerator,
    GroundedAnswer,
    SourceCitation,
    format_evidence_context,
    is_refusal_or_insufficient,
)


def make_test_chunk(
    chunk_id: str = "doc__1__0001",
    doc_id: str = "circular_170",
    source_filename: str = "Circular-170-02-2022-GST.pdf",
    page_start: int = 2,
    page_end: int = 2,
    section_path: str = "4.1",
    parent_section: str = "4",
    section_title: str = "ITC Reversal Reporting",
    body_text: str = "Reversal of input tax credit under Rule 42 is reported in Table 4(B)(1) of FORM GSTR-3B.",
    chunk_type: str = "subparagraph",
    circular_number: str = "170/02/2022-GST",
    date_issued: str = "2022-07-06",
) -> Chunk:
    """Helper to construct a test Chunk."""
    header = f"[{circular_number} | {section_path} | {section_title}]"
    full_text = f"{header}\n{body_text}"
    token_count = len(full_text.split())
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_start=page_start,
        page_end=page_end,
        section_path=section_path,
        parent_section=parent_section,
        section_title=section_title,
        text=full_text,
        token_count=token_count,
        chunk_type=chunk_type,
        parsing_method="structural",
        circular_number=circular_number,
        date_issued=date_issued,
    )


class TestGeminiGenerator(unittest.TestCase):
    """Test suite for GeminiGenerator with mock client."""

    def setUp(self):
        """Set up test fixtures and mock client."""
        self.chunk1 = make_test_chunk(
            chunk_id="c170__4_1__0001",
            section_path="4.1",
            body_text="Reversal of ITC under Rule 42 is reported in Table 4(B)(1) of FORM GSTR-3B.",
            page_start=2,
            page_end=2,
        )
        self.chunk2 = make_test_chunk(
            chunk_id="c188__5__0002",
            source_filename="circular-188.pdf",
            circular_number="188/20/2022-GST",
            section_path="5",
            section_title="Refund Procedure",
            body_text="Refund application by unregistered buyer shall be filed in FORM GST RFD-01 under Section 54(1).",
            page_start=3,
            page_end=3,
        )

        self.mock_client = MagicMock()
        self.generator = GeminiGenerator(api_key="test_key", client=self.mock_client)

    def test_1_valid_grounded_answer(self):
        """A valid model answer produces a GroundedAnswer with 'grounded' confidence label."""
        mock_response = MagicMock()
        mock_response.text = (
            "As clarified in [Circular-170-02-2022-GST.pdf, section 4.1, page 2], "
            "ITC reversal under Rule 42 is reported in Table 4(B)(1) of FORM GSTR-3B."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        results = [HybridSearchResult(chunk=self.chunk1, score=0.95, rank=1)]
        answer = self.generator.generate("Where is Rule 42 reversal reported?", results)

        self.assertIsInstance(answer, GroundedAnswer)
        self.assertEqual(answer.confidence_label, "grounded")
        self.assertIn("Table 4(B)(1)", answer.answer)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].chunk_id, "c170__4_1__0001")
        self.assertEqual(answer.used_chunk_ids, ["c170__4_1__0001"])

    def test_2_insufficient_evidence(self):
        """When the model states the answer cannot be determined, confidence is 'insufficient_evidence'."""
        mock_response = MagicMock()
        mock_response.text = "The answer cannot be determined from the provided documents as crypto taxation is not mentioned."
        self.mock_client.models.generate_content.return_value = mock_response

        results = [HybridSearchResult(chunk=self.chunk1, score=0.2, rank=1)]
        answer = self.generator.generate("What is the GST rate on cryptocurrency?", results)

        self.assertEqual(answer.confidence_label, "insufficient_evidence")
        self.assertIn("cannot be determined", answer.answer)

    def test_3_empty_retrieval_results(self):
        """Empty retrieval results return insufficient_evidence without calling Gemini."""
        answer = self.generator.generate("Any question", [])
        self.assertEqual(answer.confidence_label, "insufficient_evidence")
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.used_chunk_ids, [])
        self.assertIn("cannot be determined", answer.answer)
        # Ensure client was never called
        self.mock_client.models.generate_content.assert_not_called()

    def test_4_deterministic_source_extraction(self):
        """SourceCitation correctly extracts filename, section_path, and page spans."""
        evidence_text, sources = format_evidence_context([self.chunk1, self.chunk2])
        self.assertEqual(len(sources), 2)
        s1, s2 = sources[0], sources[1]

        self.assertEqual(s1.chunk_id, "c170__4_1__0001")
        self.assertEqual(s1.source_filename, "Circular-170-02-2022-GST.pdf")
        self.assertEqual(s1.section_path, "4.1")
        self.assertEqual(s1.page_start, 2)
        self.assertEqual(s1.page_end, 2)

        self.assertEqual(s2.chunk_id, "c188__5__0002")
        self.assertEqual(s2.source_filename, "circular-188.pdf")
        self.assertEqual(s2.section_path, "5")
        self.assertEqual(s2.page_start, 3)

    def test_5_duplicate_sources_are_deduplicated(self):
        """Duplicate chunks passed in retrieval results are deduplicated in sources list."""
        evidence_text, sources = format_evidence_context([self.chunk1, self.chunk1, self.chunk2])
        self.assertEqual(len(sources), 2)
        self.assertEqual([s.chunk_id for s in sources], ["c170__4_1__0001", "c188__5__0002"])

    def test_6_prompt_contains_the_supplied_evidence(self):
        """The prompt passed to Gemini contains the exact text of the evidence chunks."""
        mock_response = MagicMock()
        mock_response.text = "Answer text."
        self.mock_client.models.generate_content.return_value = mock_response

        _ = self.generator.generate("Test query?", [self.chunk1])

        call_args = self.mock_client.models.generate_content.call_args
        prompt_content = call_args.kwargs["contents"]

        self.assertIn("Circular-170-02-2022-GST.pdf", prompt_content)
        self.assertIn("Section Path: 4.1", prompt_content)
        self.assertIn("Table 4(B)(1)", prompt_content)

    def test_7_model_receives_the_user_question(self):
        """The exact user query is embedded into the prompt."""
        mock_response = MagicMock()
        mock_response.text = "Grounded response."
        self.mock_client.models.generate_content.return_value = mock_response

        q = "How is refund processed under Section 54?"
        _ = self.generator.generate(q, [self.chunk2])

        call_args = self.mock_client.models.generate_content.call_args
        prompt_content = call_args.kwargs["contents"]
        self.assertIn(q, prompt_content)

    def test_8_model_cannot_access_arbitrary_project_files(self):
        """Prompt is self-contained without raw file paths or open handles."""
        mock_response = MagicMock()
        mock_response.text = "Answer."
        self.mock_client.models.generate_content.return_value = mock_response

        _ = self.generator.generate("Q", [self.chunk1])
        call_args = self.mock_client.models.generate_content.call_args
        prompt_content = call_args.kwargs["contents"]

        # Only formatted context evidence and question
        self.assertTrue(prompt_content.startswith("Context Evidence:\n"))
        self.assertIn("Question:\nQ", prompt_content)

    def test_9_api_failure_is_converted_to_clean_application_error(self):
        """Gemini client exceptions are caught and raised as RuntimeError."""
        self.mock_client.models.generate_content.side_effect = Exception("API rate limit exceeded or quota error")

        with self.assertRaises(RuntimeError) as ctx:
            self.generator.generate("Query", [self.chunk1])
        self.assertIn("Gemini answer generation failed", str(ctx.exception))

    def test_10_malformed_model_response_is_handled_safely(self):
        """Empty or None text response returns insufficient_evidence gracefully."""
        mock_response = MagicMock()
        mock_response.text = None
        self.mock_client.models.generate_content.return_value = mock_response

        answer = self.generator.generate("Query", [self.chunk1])
        self.assertEqual(answer.confidence_label, "insufficient_evidence")

    def test_11_original_retrieval_objects_are_not_mutated(self):
        """Original Chunk and SearchResult objects are not altered during generation."""
        orig_dict = asdict(self.chunk1)
        mock_response = MagicMock()
        mock_response.text = "Valid answer."
        self.mock_client.models.generate_content.return_value = mock_response

        res = [HybridSearchResult(chunk=self.chunk1, score=0.8, rank=1)]
        _ = self.generator.generate("Query", res)

        self.assertEqual(asdict(self.chunk1), orig_dict)


if __name__ == "__main__":
    unittest.main()
