"""
Unit tests for DenseRetriever in retrieval/dense_retriever.py.
Covers index building, vector dimensionality, semantic search ranking,
top_k limits, empty query/corpus handling, determinism, persistence,
duplicate text handling, and metadata preservation.
"""

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np

from ingestion.chunker import Chunk
from retrieval.dense_retriever import DenseRetriever, SearchResult, DENSE_INDEX_PATH
from config.settings import EMBEDDING_DIM, EMBEDDING_MODEL


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


class TestDenseRetriever(unittest.TestCase):
    """Test suite for DenseRetriever."""

    def setUp(self):
        """Create a synthetic GST corpus with precomputed orthogonal/directional embeddings."""
        self.corpus = [
            make_test_chunk(
                chunk_id="c170__3_1__0001",
                doc_id="Circular-170",
                source_filename="Circular-170-02-2022-GST.pdf",
                section_path="3.1",
                section_title="Inter-State Supplies",
                body_text="Furnishing of information regarding inter-State supplies to unregistered persons in FORM GSTR-3B.",
            ),
            make_test_chunk(
                chunk_id="c170__4_1__0002",
                doc_id="Circular-170",
                source_filename="Circular-170-02-2022-GST.pdf",
                section_path="4.1",
                section_title="ITC Reversal",
                body_text="Manner of reporting ITC reversal under Rule 42 and Rule 43 of CGST Rules.",
            ),
            make_test_chunk(
                chunk_id="c172__5__0003",
                doc_id="Circular-172",
                source_filename="Circular-172-04-2022-GST.pdf",
                section_path="5",
                section_title="Perquisites to Employees",
                body_text="Perquisites provided by employer to employees as per contractual agreement are not taxable.",
            ),
            make_test_chunk(
                chunk_id="c188__5__0004",
                doc_id="Circular-188",
                source_filename="circular-188.pdf",
                section_path="5",
                section_title="Refund Application",
                body_text="Relevant date and procedure for filing refund application by unregistered buyers under Section 54(1).",
            ),
        ]

        # Construct deterministic normalized embeddings
        dim = EMBEDDING_DIM  # 384
        np.random.seed(42)
        raw_vecs = np.random.randn(len(self.corpus), dim).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        self.mock_embeddings = raw_vecs / norms

        self.retriever = DenseRetriever(
            chunks=self.corpus,
            embeddings=self.mock_embeddings,
            dimension=dim,
            normalize=True,
        )

    def test_a_index_builds_successfully(self):
        """Index initializes properly with matching corpus size and embedding dimensions."""
        self.assertEqual(self.retriever.corpus_size, 4)
        self.assertEqual(self.retriever.embeddings.shape, (4, EMBEDDING_DIM))

    def test_b_embeddings_have_expected_dimensionality(self):
        """All vector rows match EMBEDDING_DIM (384)."""
        self.assertEqual(self.retriever.embeddings.shape[1], EMBEDDING_DIM)

    def test_c_semantically_similar_query_ranks_appropriate_chunk_highly(self):
        """A query aligned with chunk 2 vector scores highest on chunk 2."""
        # Query vector close to chunk 2 (index 1: ITC reversal)
        query_vec = self.mock_embeddings[1].copy()
        # Mock encode query method for this test
        self.retriever._encode_query = lambda q: query_vec

        results = self.retriever.search("How to report ITC reversal under Rule 42?", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.chunk_id, "c170__4_1__0002")
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 1.0, places=5)

    def test_d_top_k_is_respected(self):
        """Returned results length never exceeds top_k."""
        self.retriever._encode_query = lambda q: self.mock_embeddings[0]
        results = self.retriever.search("test query", top_k=2)
        self.assertEqual(len(results), 2)

    def test_e_empty_query_handled_predictably(self):
        """Empty, whitespace, or non-string query returns empty list."""
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("   "), [])
        self.assertEqual(self.retriever.search(None), [])  # type: ignore

    def test_f_empty_corpus_handled_predictably(self):
        """Retriever with zero chunks returns empty list without error."""
        empty_retriever = DenseRetriever(chunks=[])
        self.assertEqual(empty_retriever.corpus_size, 0)
        self.assertEqual(empty_retriever.embeddings.shape, (0, EMBEDDING_DIM))
        self.assertEqual(empty_retriever.search("ITC"), [])

    def test_g_repeated_query_produces_identical_results(self):
        """Repeated searches produce bit-exact identical ranks and similarity scores."""
        self.retriever._encode_query = lambda q: self.mock_embeddings[2]
        res1 = self.retriever.search("Perquisites query", top_k=4)
        res2 = self.retriever.search("Perquisites query", top_k=4)
        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.chunk.chunk_id, r2.chunk.chunk_id)
            self.assertEqual(r1.score, r2.score)
            self.assertEqual(r1.rank, r2.rank)

    def test_h_persistence_save_load_preserves_ranking(self):
        """Saved and restored DenseRetriever produces identical search results."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.retriever.save(tmp_path)
            loaded = DenseRetriever.load(tmp_path)

            self.assertEqual(loaded.corpus_size, self.retriever.corpus_size)
            self.assertEqual(loaded.embeddings.shape, self.retriever.embeddings.shape)

            # Test search on loaded instance
            query_vec = self.mock_embeddings[3]
            loaded._encode_query = lambda q: query_vec
            self.retriever._encode_query = lambda q: query_vec

            res_orig = self.retriever.search("refund", top_k=4)
            res_loaded = loaded.search("refund", top_k=4)

            self.assertEqual(len(res_orig), len(res_loaded))
            for r_orig, r_loaded in zip(res_orig, res_loaded):
                self.assertEqual(r_orig.chunk.chunk_id, r_loaded.chunk.chunk_id)
                self.assertAlmostEqual(r_orig.score, r_loaded.score, places=6)
                self.assertEqual(r_orig.rank, r_loaded.rank)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_i_duplicate_text_remains_distinguishable_by_chunk_id(self):
        """Duplicate text chunks with identical vectors are tie-broken deterministically by chunk_id."""
        c1 = make_test_chunk(chunk_id="doc__1__0001", body_text="Duplicate body.")
        c2 = make_test_chunk(chunk_id="doc__1__0002", body_text="Duplicate body.")
        vec = self.mock_embeddings[0]
        dup_embeddings = np.array([vec, vec], dtype=np.float32)

        retriever = DenseRetriever(chunks=[c2, c1], embeddings=dup_embeddings)  # reverse order
        retriever._encode_query = lambda q: vec

        results = retriever.search("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].score, results[1].score, places=6)
        # Tie-breaker: doc__1__0001 before doc__1__0002
        self.assertEqual(results[0].chunk.chunk_id, "doc__1__0001")
        self.assertEqual(results[1].chunk.chunk_id, "doc__1__0002")

    def test_j_metadata_preserved(self):
        """Returned Chunk objects retain all metadata fields unaltered."""
        self.retriever._encode_query = lambda q: self.mock_embeddings[2]
        results = self.retriever.search("query", top_k=1)
        self.assertEqual(len(results), 1)
        c = results[0].chunk
        self.assertEqual(c.doc_id, "Circular-172")
        self.assertEqual(c.source_filename, "Circular-172-04-2022-GST.pdf")
        self.assertEqual(c.section_path, "5")
        self.assertEqual(c.section_title, "Perquisites to Employees")

    def test_k_original_chunk_objects_are_not_mutated(self):
        """Search does not modify the original Chunk objects in the corpus."""
        orig_dict_before = [asdict(c) for c in self.corpus]
        self.retriever._encode_query = lambda q: self.mock_embeddings[0]
        _ = self.retriever.search("any query", top_k=4)
        orig_dict_after = [asdict(c) for c in self.corpus]
        self.assertEqual(orig_dict_before, orig_dict_after)

    def test_l_embedding_matrix_size_matches_chunk_count(self):
        """The embedding matrix rows equal corpus_size."""
        self.assertEqual(self.retriever.embeddings.shape[0], len(self.corpus))
        self.assertEqual(self.retriever.embeddings.shape[1], EMBEDDING_DIM)

    def test_m_end_to_end_real_model_embedding(self):
        """Builds DenseRetriever using the real SentenceTransformer model and verifies semantic retrieval."""
        real_retriever = DenseRetriever.build(self.corpus[:3])
        self.assertEqual(real_retriever.corpus_size, 3)
        self.assertEqual(real_retriever.embeddings.shape, (3, EMBEDDING_DIM))

        # Semantic search without mock
        results = real_retriever.search("reporting input tax credit reversal under Rule 42", top_k=2)
        self.assertGreater(len(results), 0)
        # Should retrieve c170__4_1__0002 (ITC reversal) at rank 1
        self.assertEqual(results[0].chunk.chunk_id, "c170__4_1__0002")
        self.assertEqual(results[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
