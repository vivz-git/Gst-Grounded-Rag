"""
Unit tests for BM25Retriever in retrieval/bm25_retriever.py.
Covers index building, keyword matching, rare term weighting, circular number queries,
section/rule citations, top_k limits, empty query/corpus handling, determinism,
persistence save/load fidelity, duplicate text handling, and metadata preservation.
"""

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever, SearchResult, tokenize


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


class TestBM25Retriever(unittest.TestCase):
    """Test suite for BM25 lexical retrieval."""

    def setUp(self):
        """Create a standard synthetic GST corpus fixture."""
        self.corpus = [
            make_test_chunk(
                chunk_id="circ170__3_1__0001",
                doc_id="Circular-170",
                source_filename="Circular-170-02-2022-GST.pdf",
                circular_number="170/02/2022-GST",
                section_path="3.1",
                section_title="Inter-State Supplies",
                body_text="Furnishing of information regarding inter-State supplies made to unregistered persons in FORM GSTR-3B and Table 3.2.",
            ),
            make_test_chunk(
                chunk_id="circ170__4_1__0002",
                doc_id="Circular-170",
                source_filename="Circular-170-02-2022-GST.pdf",
                circular_number="170/02/2022-GST",
                section_path="4.1",
                section_title="ITC Reversal",
                body_text="Manner of reporting ITC reversal under Rule 42 and Rule 43 of CGST Rules in Table 4(B) of FORM GSTR-3B.",
            ),
            make_test_chunk(
                chunk_id="circ172__5__0003",
                doc_id="Circular-172",
                source_filename="Circular-172-04-2022-GST.pdf",
                circular_number="172/04/2022-GST",
                section_path="5",
                section_title="Perquisites to Employees",
                body_text="Perquisites provided by employer to employees as per contractual agreement are not subject to GST under Section 17(5).",
            ),
            make_test_chunk(
                chunk_id="circ183__4_1__0004",
                doc_id="Circular-183",
                source_filename="circular-183.pdf",
                circular_number="183/15/2022-GST",
                section_path="4.1",
                section_title="Verification of ITC",
                body_text="Verification of condition of clause (c) of sub-section (2) of Section 16 of CGST Act where difference is upto Rs. 5 lakh under Rule 36(4).",
            ),
            make_test_chunk(
                chunk_id="circ188__5__0005",
                doc_id="Circular-188",
                source_filename="circular-188.pdf",
                circular_number="188/20/2022-GST",
                section_path="5",
                section_title="Refund Application",
                body_text="Relevant date and filing of refund application by unregistered buyers under Section 54(1) for cancelled construction contracts.",
            ),
        ]
        self.retriever = BM25Retriever.build(self.corpus)

    def test_a_index_builds_from_real_chunk_objects(self):
        """BM25 index initializes correctly with corpus size, doc lengths, and IDFs."""
        self.assertEqual(self.retriever.corpus_size, 5)
        self.assertEqual(len(self.retriever.doc_lengths), 5)
        self.assertGreater(self.retriever.avgdl, 0)
        self.assertIn("itc", self.retriever.idf)
        self.assertIn("gstr-3b", self.retriever.idf)

    def test_b_exact_keyword_match_ranks_highly(self):
        """A specific keyword query retrieves the exact matching chunk at rank 1."""
        results = self.retriever.search("perquisites employer contractual agreement", top_k=3)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result.chunk.chunk_id, "circ172__5__0003")
        self.assertEqual(top_result.rank, 1)
        self.assertIn("Perquisites", top_result.chunk.text)

    def test_c_rare_legal_term_beats_generic_term(self):
        """Rare terms (high IDF) have higher weight than ubiquitous generic terms."""
        results = self.retriever.search("construction refund application", top_k=3)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result.chunk.chunk_id, "circ188__5__0005")
        self.assertEqual(top_result.rank, 1)

    def test_d_circular_number_queries_work(self):
        """Exact circular number search matches the target circular chunk."""
        results = self.retriever.search("183/15/2022-GST", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.circular_number, "183/15/2022-GST")

    def test_e_rule_and_section_references_work(self):
        """Citations like 'Rule 42' or 'Section 17(5)' retrieve the right chunks."""
        # Rule 42
        r_rule42 = self.retriever.search("Rule 42", top_k=2)
        self.assertGreater(len(r_rule42), 0)
        self.assertEqual(r_rule42[0].chunk.chunk_id, "circ170__4_1__0002")

        # Section 17(5)
        r_sec17 = self.retriever.search("Section 17(5)", top_k=2)
        self.assertGreater(len(r_sec17), 0)
        self.assertEqual(r_sec17[0].chunk.chunk_id, "circ172__5__0003")

    def test_f_top_k_is_respected(self):
        """The number of returned results never exceeds the requested top_k."""
        results = self.retriever.search("GST", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_g_empty_query_handled_predictably(self):
        """Empty, whitespace, or non-string query returns an empty list."""
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("   "), [])
        self.assertEqual(self.retriever.search(None), [])  # type: ignore

    def test_h_empty_corpus_handled_predictably(self):
        """An empty retriever returns empty list for any search query."""
        empty_retriever = BM25Retriever.build([])
        self.assertEqual(empty_retriever.corpus_size, 0)
        self.assertEqual(empty_retriever.search("ITC"), [])

    def test_i_deterministic_ranking_across_repeated_searches(self):
        """Repeated searches with the same query produce identical scores and ranks."""
        res1 = self.retriever.search("ITC reversal Rule 42 GSTR-3B", top_k=5)
        res2 = self.retriever.search("ITC reversal Rule 42 GSTR-3B", top_k=5)
        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.chunk.chunk_id, r2.chunk.chunk_id)
            self.assertEqual(r1.score, r2.score)
            self.assertEqual(r1.rank, r2.rank)

    def test_j_persistence_save_and_load_produces_identical_results(self):
        """Saving and loading the BM25 index produces bit-exact search scores and ranks."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.retriever.save(tmp_path)
            loaded_retriever = BM25Retriever.load(tmp_path)

            self.assertEqual(loaded_retriever.corpus_size, self.retriever.corpus_size)

            q = "Manner of reporting ITC reversal under Rule 42"
            res_orig = self.retriever.search(q, top_k=5)
            res_loaded = loaded_retriever.search(q, top_k=5)

            self.assertEqual(len(res_orig), len(res_loaded))
            for r_orig, r_loaded in zip(res_orig, res_loaded):
                self.assertEqual(r_orig.chunk.chunk_id, r_loaded.chunk.chunk_id)
                self.assertAlmostEqual(r_orig.score, r_loaded.score, places=7)
                self.assertEqual(r_orig.rank, r_loaded.rank)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_k_duplicate_text_remains_distinguishable_by_chunk_id(self):
        """Identical text chunks are distinguished by chunk_id tie-breaking."""
        dup1 = make_test_chunk(chunk_id="doc__1__0001", body_text="Exact duplicate text.")
        dup2 = make_test_chunk(chunk_id="doc__1__0002", body_text="Exact duplicate text.")
        retriever = BM25Retriever.build([dup2, dup1])  # passed in reverse order

        results = retriever.search("duplicate text", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, results[1].score)
        # Tie-breaker sorts by chunk_id ascending: doc__1__0001 before doc__1__0002
        self.assertEqual(results[0].chunk.chunk_id, "doc__1__0001")
        self.assertEqual(results[1].chunk.chunk_id, "doc__1__0002")

    def test_l_metadata_is_preserved(self):
        """All metadata fields on returned chunk objects remain intact."""
        results = self.retriever.search("Rule 36(4)", top_k=1)
        self.assertEqual(len(results), 1)
        c = results[0].chunk
        self.assertEqual(c.doc_id, "Circular-183")
        self.assertEqual(c.source_filename, "circular-183.pdf")
        self.assertEqual(c.circular_number, "183/15/2022-GST")
        self.assertEqual(c.section_path, "4.1")
        self.assertEqual(c.section_title, "Verification of ITC")

    def test_m_original_chunk_objects_are_not_mutated(self):
        """Original chunk objects passed to build() are not altered by search."""
        orig_dict_before = [asdict(c) for c in self.corpus]
        _ = self.retriever.search("ITC reversal", top_k=5)
        orig_dict_after = [asdict(c) for c in self.corpus]
        self.assertEqual(orig_dict_before, orig_dict_after)


if __name__ == "__main__":
    unittest.main()
