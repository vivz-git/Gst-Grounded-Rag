"""
Unit tests for HybridRetriever in retrieval/hybrid_retriever.py.
Covers score normalization, BM25-only/dense-only/dual matches, deduplication,
weight customization, deterministic tie-breaking, top_k limits, metadata integrity,
empty query/corpus handling, and retrieval source tracking.
"""

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np

from config.settings import BM25_WEIGHT, DENSE_WEIGHT, EMBEDDING_DIM
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever, SearchResult
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    normalize_scores,
)


def make_test_chunk(
    chunk_id: str = "doc__1__0001",
    doc_id: str = "circular_172",
    source_filename: str = "Circular-172-04-2022-GST.pdf",
    page_start: int = 1,
    page_end: int = 1,
    section_path: str = "1",
    parent_section: str = "",
    section_title: str = "General Provisions",
    body_text: str = "Sample chunk body text.",
    chunk_type: str = "paragraph",
    circular_number: str = "172/04/2022-GST",
    date_issued: str = "2022-07-06",
) -> Chunk:
    """Helper to construct test Chunk with context header."""
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


class TestHybridRetriever(unittest.TestCase):
    """Test suite for HybridRetriever."""

    def setUp(self):
        """Create deterministic fixture chunks and retrievers."""
        self.c1 = make_test_chunk(
            chunk_id="chunk_1",
            body_text="Rule 42 input tax credit reversal computation instructions.",
            section_path="4.1",
        )
        self.c2 = make_test_chunk(
            chunk_id="chunk_2",
            body_text="Refund application filing for cancelled flat purchases under Section 54.",
            section_path="5",
        )
        self.c3 = make_test_chunk(
            chunk_id="chunk_3",
            body_text="Perquisites to employees pursuant to employment contract not taxable.",
            section_path="6",
        )
        self.corpus = [self.c1, self.c2, self.c3]

        # Synthetic BM25 and Dense instances with deterministic mocks
        self.mock_bm25 = MagicMock(spec=BM25Retriever)
        self.mock_bm25.corpus_size = 3

        self.mock_dense = MagicMock(spec=DenseRetriever)
        self.mock_dense.corpus_size = 3

        self.hybrid = HybridRetriever(
            bm25_retriever=self.mock_bm25,
            dense_retriever=self.mock_dense,
            bm25_weight=0.5,
            dense_weight=0.5,
        )

    def test_1_bm25_only_match(self):
        """Chunk returned only by BM25 receives 0.0 dense contribution."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=10.0, rank=1)
        ]
        self.mock_dense.search.return_value = []

        results = self.hybrid.search("Rule 42")
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.chunk.chunk_id, "chunk_1")
        self.assertEqual(res.retrieval_sources, ["bm25"])
        self.assertEqual(res.bm25_score, 10.0)
        self.assertIsNone(res.dense_score)
        self.assertEqual(res.normalized_bm25_score, 1.0)
        self.assertEqual(res.normalized_dense_score, 0.0)
        self.assertEqual(res.score, 0.5 * 1.0 + 0.5 * 0.0)

    def test_2_dense_only_match(self):
        """Chunk returned only by dense receives 0.0 BM25 contribution."""
        self.mock_bm25.search.return_value = []
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.85, rank=1)
        ]

        results = self.hybrid.search("refund procedure")
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.chunk.chunk_id, "chunk_2")
        self.assertEqual(res.retrieval_sources, ["dense"])
        self.assertIsNone(res.bm25_score)
        self.assertEqual(res.dense_score, 0.85)
        self.assertEqual(res.normalized_bm25_score, 0.0)
        self.assertEqual(res.normalized_dense_score, 1.0)
        self.assertEqual(res.score, 0.5 * 0.0 + 0.5 * 1.0)

    def test_3_chunk_appearing_in_both_retrievers_is_deduplicated(self):
        """A chunk returned by both BM25 and dense appears exactly once in the final list."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=12.0, rank=1)
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c1, score=0.90, rank=1)
        ]

        results = self.hybrid.search("Rule 42")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "chunk_1")
        self.assertEqual(results[0].retrieval_sources, ["bm25", "dense"])

    def test_4_combined_score_is_calculated_correctly(self):
        """Combined score matches weighted sum of normalized scores."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=20.0, rank=1),
            SearchResult(chunk=self.c2, score=10.0, rank=2),
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.80, rank=1),
            SearchResult(chunk=self.c1, score=0.40, rank=2),
        ]
        # BM25: c1 -> (20-10)/(20-10) = 1.0, c2 -> (10-10)/10 = 0.0
        # Dense: c2 -> (0.8-0.4)/(0.8-0.4) = 1.0, c1 -> (0.4-0.4)/0.4 = 0.0
        # Combined c1 = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # Combined c2 = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        results = self.hybrid.search("tax query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, 0.5)
        self.assertEqual(results[1].score, 0.5)

    def test_5_default_weights_are_05_05(self):
        """Default weights match settings BM25_WEIGHT (0.5) and DENSE_WEIGHT (0.5)."""
        retriever = HybridRetriever(self.mock_bm25, self.mock_dense)
        self.assertEqual(retriever.bm25_weight, BM25_WEIGHT)
        self.assertEqual(retriever.dense_weight, DENSE_WEIGHT)
        self.assertEqual(retriever.bm25_weight, 0.5)
        self.assertEqual(retriever.dense_weight, 0.5)

    def test_6_custom_weights_work(self):
        """Custom weights alter final ranking according to priority."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=20.0, rank=1),  # norm = 1.0
            SearchResult(chunk=self.c2, score=10.0, rank=2),  # norm = 0.0
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.90, rank=1),  # norm = 1.0
            SearchResult(chunk=self.c1, score=0.30, rank=2),  # norm = 0.0
        ]

        # Case A: Heavy BM25 weight (0.8 / 0.2)
        h_bm25_heavy = HybridRetriever(self.mock_bm25, self.mock_dense, bm25_weight=0.8, dense_weight=0.2)
        res_a = h_bm25_heavy.search("query")
        # c1 score: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        # c2 score: 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        self.assertEqual(res_a[0].chunk.chunk_id, "chunk_1")
        self.assertEqual(res_a[0].score, 0.8)

        # Case B: Heavy Dense weight (0.2 / 0.8)
        h_dense_heavy = HybridRetriever(self.mock_bm25, self.mock_dense, bm25_weight=0.2, dense_weight=0.8)
        res_b = h_dense_heavy.search("query")
        # c2 score: 0.2 * 0.0 + 0.8 * 1.0 = 0.8
        # c1 score: 0.2 * 1.0 + 0.8 * 0.0 = 0.2
        self.assertEqual(res_b[0].chunk.chunk_id, "chunk_2")
        self.assertEqual(res_b[0].score, 0.8)

    def test_7_min_max_normalization_works(self):
        """Helper normalize_scores maps arbitrary positive ranges to [0.0, 1.0]."""
        raw = {"c1": 100.0, "c2": 50.0, "c3": 0.0}
        norm = normalize_scores(raw)
        self.assertAlmostEqual(norm["c1"], 1.0)
        self.assertAlmostEqual(norm["c2"], 0.5)
        self.assertAlmostEqual(norm["c3"], 0.0)

    def test_8_identical_scores_do_not_cause_division_by_zero(self):
        """Equal scores across candidates normalize cleanly to 1.0 without ZeroDivisionError."""
        identical = {"c1": 5.5, "c2": 5.5, "c3": 5.5}
        norm = normalize_scores(identical)
        self.assertEqual(norm["c1"], 1.0)
        self.assertEqual(norm["c2"], 1.0)
        self.assertEqual(norm["c3"], 1.0)

        # Single item
        single = {"c1": 12.0}
        norm_single = normalize_scores(single)
        self.assertEqual(norm_single["c1"], 1.0)

        # All zeros
        all_zeros = {"c1": 0.0, "c2": 0.0}
        norm_zeros = normalize_scores(all_zeros)
        self.assertEqual(norm_zeros["c1"], 0.0)
        self.assertEqual(norm_zeros["c2"], 0.0)

    def test_9_deterministic_tie_breaking(self):
        """When combined scores are equal, chunk_id ascending determines rank order."""
        # Both chunks have score 0.5
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c2, score=10.0, rank=1),  # chunk_2
            SearchResult(chunk=self.c1, score=10.0, rank=2),  # chunk_1
        ]
        self.mock_dense.search.return_value = []

        results = self.hybrid.search("query")
        self.assertEqual(len(results), 2)
        # Tie-breaker: chunk_1 before chunk_2
        self.assertEqual(results[0].chunk.chunk_id, "chunk_1")
        self.assertEqual(results[1].chunk.chunk_id, "chunk_2")

    def test_10_top_k_is_respected(self):
        """Hybrid search returns at most top_k items."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=10.0, rank=1),
            SearchResult(chunk=self.c2, score=8.0, rank=2),
            SearchResult(chunk=self.c3, score=6.0, rank=3),
        ]
        self.mock_dense.search.return_value = []

        results = self.hybrid.search("query", top_k=2)
        self.assertEqual(len(results), 2)

    def test_11_metadata_is_preserved(self):
        """All metadata fields on returned Chunk objects remain accessible and unmodified."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=10.0, rank=1)
        ]
        self.mock_dense.search.return_value = []

        results = self.hybrid.search("Rule 42", top_k=1)
        self.assertEqual(len(results), 1)
        c = results[0].chunk
        self.assertEqual(c.doc_id, "circular_172")
        self.assertEqual(c.source_filename, "Circular-172-04-2022-GST.pdf")
        self.assertEqual(c.circular_number, "172/04/2022-GST")
        self.assertEqual(c.section_path, "4.1")

    def test_12_original_chunk_objects_are_not_modified(self):
        """Chunks in the corpus are not mutated by hybrid search."""
        orig_dict_before = [asdict(c) for c in self.corpus]
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=10.0, rank=1)
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.8, rank=1)
        ]

        _ = self.hybrid.search("query", top_k=3)
        orig_dict_after = [asdict(c) for c in self.corpus]
        self.assertEqual(orig_dict_before, orig_dict_after)

    def test_13_empty_query_handled_predictably(self):
        """Empty or whitespace query returns empty list without error."""
        self.assertEqual(self.hybrid.search(""), [])
        self.assertEqual(self.hybrid.search("   "), [])
        self.assertEqual(self.hybrid.search(None), [])  # type: ignore

    def test_14_empty_corpus_handled_predictably(self):
        """Retriever with empty underlying retrievers returns empty list."""
        mock_empty_bm25 = MagicMock(spec=BM25Retriever)
        mock_empty_bm25.corpus_size = 0
        mock_empty_bm25.search.return_value = []

        mock_empty_dense = MagicMock(spec=DenseRetriever)
        mock_empty_dense.corpus_size = 0
        mock_empty_dense.search.return_value = []

        empty_hybrid = HybridRetriever(mock_empty_bm25, mock_empty_dense)
        self.assertEqual(empty_hybrid.search("any query"), [])

    def test_15_repeated_query_gives_identical_results(self):
        """Repeated hybrid searches produce deterministic, identical result ordering and scores."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=15.0, rank=1),
            SearchResult(chunk=self.c2, score=10.0, rank=2),
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.85, rank=1),
            SearchResult(chunk=self.c3, score=0.60, rank=2),
        ]

        run1 = self.hybrid.search("test query", top_k=3)
        run2 = self.hybrid.search("test query", top_k=3)

        self.assertEqual(len(run1), len(run2))
        for r1, r2 in zip(run1, run2):
            self.assertEqual(r1.chunk.chunk_id, r2.chunk.chunk_id)
            self.assertEqual(r1.score, r2.score)
            self.assertEqual(r1.rank, r2.rank)
            self.assertEqual(r1.retrieval_sources, r2.retrieval_sources)

    def test_16_retrieval_sources_correctly_reports_bm25_vs_dense_vs_both(self):
        """Verification of retrieval_sources classification."""
        self.mock_bm25.search.return_value = [
            SearchResult(chunk=self.c1, score=10.0, rank=1),
            SearchResult(chunk=self.c2, score=5.0, rank=2),
        ]
        self.mock_dense.search.return_value = [
            SearchResult(chunk=self.c2, score=0.9, rank=1),
            SearchResult(chunk=self.c3, score=0.7, rank=2),
        ]

        results = self.hybrid.search("test", top_k=5)
        res_by_id = {r.chunk.chunk_id: r for r in results}

        # c1 -> BM25 only
        self.assertEqual(res_by_id["chunk_1"].retrieval_sources, ["bm25"])
        # c2 -> both
        self.assertEqual(res_by_id["chunk_2"].retrieval_sources, ["bm25", "dense"])
        # c3 -> dense only
        self.assertEqual(res_by_id["chunk_3"].retrieval_sources, ["dense"])


if __name__ == "__main__":
    unittest.main()
