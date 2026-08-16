"""
Unit tests for chunk_document() orchestration in ingestion/chunker.py.
Covers empty documents, single/multi-page flows, cross-page continuations,
page order stability, single end flush, fragment merging, oversized splitting,
table atomicity, metadata propagation, input immutability, size policy enforcement,
and input validation errors.
"""

import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.pdf_parser import PageText, ParsedDocument
from ingestion.chunker import (
    Chunk,
    chunk_document,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
)


def make_doc_metadata(
    doc_id: str = "circular_172",
    source_filename: str = "Circular-172-04-2022-GST.pdf",
    circular_number: str = "172/04/2022-GST",
    date_issued: str = "2022-07-06",
) -> dict:
    """Helper to construct valid doc_metadata dictionary."""
    return {
        "doc_id": doc_id,
        "source_filename": source_filename,
        "circular_number": circular_number,
        "date_issued": date_issued,
    }


def make_parsed_doc(
    pages: list[PageText] = None,
    file_name: str = "Circular-172-04-2022-GST.pdf",
) -> ParsedDocument:
    """Helper to construct a ParsedDocument fixture."""
    pages = pages or []
    full_text = "\n\n".join(p.text for p in pages)
    return ParsedDocument(
        file_path=f"data/pdfs/{file_name}",
        file_name=file_name,
        page_count=len(pages),
        pages=pages,
        raw_text=full_text,
        total_chars=len(full_text),
    )


class TestChunkDocument(unittest.TestCase):
    """Test suite for end-to-end document chunking orchestration."""

    def test_1_empty_parsed_document(self):
        """Empty ParsedDocument with zero pages returns an empty list."""
        doc = make_parsed_doc(pages=[])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(chunks, [])

    def test_2_single_page_document(self):
        """Single-page document produces structured chunks for each paragraph."""
        p1 = PageText(
            page_number=1,
            text=(
                "Circular No. 172/04/2022-GST\n"
                "To, The Pr. Chief Commissioners of Central Tax\n"
                "1. Various representations have been received requesting clarification on GST rates.\n"
                "2. The issues have been examined and clarified as follows for trade ease.\n"
            ),
            raw_text="...",
            char_count=200,
        )
        doc = make_parsed_doc(pages=[p1])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertGreaterEqual(len(chunks), 2)
        paths = [c.section_path for c in chunks]
        self.assertIn("1", paths)
        self.assertIn("2", paths)
        for c in chunks:
            self.assertEqual(c.page_start, 1)
            self.assertEqual(c.page_end, 1)

    def test_3_multi_page_document(self):
        """Multi-page document produces chunks spanning multiple pages."""
        p1 = PageText(
            page_number=1,
            text="1. First paragraph on page one of the circular with complete text explanation.\n",
            raw_text="...",
            char_count=80,
        )
        p2 = PageText(
            page_number=2,
            text="2. Second paragraph on page two with comprehensive procedural instructions.\n",
            raw_text="...",
            char_count=80,
        )
        p3 = PageText(
            page_number=3,
            text="3. Third paragraph on page three covering administrative jurisdiction.\n",
            raw_text="...",
            char_count=80,
        )
        doc = make_parsed_doc(pages=[p1, p2, p3])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[1].page_start, 2)
        self.assertEqual(chunks[2].page_start, 3)

    def test_4_cross_page_paragraph_continuation(self):
        """Paragraph starting on page 1 and continuing on page 2 creates a single multi-page chunk."""
        p1 = PageText(
            page_number=1,
            text="1. The registered person is eligible to claim input tax credit provided conditions are met,\n",
            raw_text="...",
            char_count=90,
        )
        p2 = PageText(
            page_number=2,
            text="including payment of tax to the Government and filing of FORM GSTR-3B on time.\n",
            raw_text="...",
            char_count=85,
        )
        doc = make_parsed_doc(pages=[p1, p2])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_path, "1")
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[0].page_end, 2)
        self.assertIn("provided conditions are met", chunks[0].text)
        self.assertIn("payment of tax to the Government", chunks[0].text)

    def test_5_page_order_is_preserved(self):
        """Chunks appear in strict ascending document and page order."""
        pages = [
            PageText(page_number=i, text=f"{i}. Paragraph number {i} text on page {i}.\n", raw_text="...", char_count=50)
            for i in range(1, 6)
        ]
        doc = make_parsed_doc(pages=pages)
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 5)
        self.assertEqual([c.section_path for c in chunks], ["1", "2", "3", "4", "5"])
        self.assertEqual([c.page_start for c in chunks], [1, 2, 3, 4, 5])

    def test_6_final_state_is_flushed_exactly_once(self):
        """The last paragraph of the final page is flushed and included in output."""
        p1 = PageText(page_number=1, text="1. First para text.\n", raw_text="...", char_count=30)
        p2 = PageText(page_number=2, text="9. Difficulty, if any, in implementation may be brought to notice.\n", raw_text="...", char_count=70)
        doc = make_parsed_doc(pages=[p1, p2])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1].section_path, "9")
        self.assertIn("Difficulty, if any", chunks[-1].text)

    def test_7_small_fragments_are_merged(self):
        """Sub-threshold fragments (< MIN_CHUNK_TOKENS) are merged into compatible chunks."""
        p1 = PageText(
            page_number=1,
            text=(
                "4.1 Procedure.\n"  # 2 words -> tiny fragment
                "4.1.1 First sub-rule text: " + " ".join(["rule explanation text"] * 15) + ".\n"
            ),
            raw_text="...",
            char_count=200,
        )
        doc = make_parsed_doc(pages=[p1])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_path, "4.1.1")
        self.assertIn("4.1 Procedure", chunks[0].text)

    def test_8_oversized_chunks_are_split(self):
        """Oversized non-table chunks (> MAX_CHUNK_TOKENS) are split at legal boundaries."""
        s1 = "In terms of Section 16 of the CGST Act, 2017, verification rules apply. " + " ".join(["detail"] * 600) + "."
        s2 = "Furthermore, under Rule 36(4) of the CGST Rules, reconciliation is mandatory. " + " ".join(["detail"] * 600) + "."
        p1 = PageText(
            page_number=1,
            text=f"1. {s1}\n{s2}\n",
            raw_text="...",
            char_count=3000,
        )
        doc = make_parsed_doc(pages=[p1])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, MAX_CHUNK_TOKENS)

    def test_9_table_pages_remain_atomic(self):
        """Table pages produce atomic table chunks and are neither split nor merged into normal text."""
        p1 = PageText(
            page_number=1,
            text="1. Normal introductory paragraph before table.\n",
            raw_text="...",
            char_count=50,
        )
        p2 = PageText(
            page_number=2,
            text="Whether ITC is allowed? " + " ".join(["Question answer row content"] * 100),
            raw_text="...",
            char_count=600,
            has_table=True,
        )
        p3 = PageText(
            page_number=3,
            text="3. Closing paragraph after table.\n",
            raw_text="...",
            char_count=40,
        )
        doc = make_parsed_doc(pages=[p1, p2, p3])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        table_chunks = [c for c in chunks if c.chunk_type.startswith("table")]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0].chunk_type, "table_qa")
        self.assertEqual(table_chunks[0].page_start, 2)
        self.assertEqual(table_chunks[0].page_end, 2)

    def test_10_metadata_is_preserved(self):
        """All chunks retain exact metadata passed through doc_metadata."""
        p1 = PageText(page_number=1, text="1. Paragraph one text.\n", raw_text="...", char_count=30)
        doc = make_parsed_doc(pages=[p1], file_name="Custom-Circular.pdf")
        meta = {
            "doc_id": "custom_circ_99",
            "source_filename": "Custom-Circular.pdf",
            "circular_number": "99/01/2023-GST",
            "date_issued": "2023-01-15",
        }
        chunks = chunk_document(doc, meta)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.doc_id, "custom_circ_99")
        self.assertEqual(c.source_filename, "Custom-Circular.pdf")
        self.assertEqual(c.circular_number, "99/01/2023-GST")
        self.assertEqual(c.date_issued, "2023-01-15")

    def test_11_chunk_ordering_is_deterministic_across_two_identical_runs(self):
        """Calling chunk_document() twice on the same input produces identical output."""
        pages = [
            PageText(page_number=1, text="1. Para one body.\n2. Para two body.\n", raw_text="...", char_count=60),
            PageText(page_number=2, text="3. Para three body.\n", raw_text="...", char_count=30),
        ]
        doc = make_parsed_doc(pages=pages)
        meta = make_doc_metadata()

        run1 = chunk_document(doc, meta)
        run2 = chunk_document(doc, meta)

        self.assertEqual(len(run1), len(run2))
        self.assertEqual([asdict(c) for c in run1], [asdict(c) for c in run2])

    def test_12_input_parsed_document_remains_unchanged(self):
        """Input ParsedDocument and PageText objects are not mutated during chunking."""
        p1 = PageText(page_number=1, text="1. Sample text for immutability.\n", raw_text="raw1", char_count=35)
        p2 = PageText(page_number=2, text="2. Second page sample text.\n", raw_text="raw2", char_count=30)
        doc = make_parsed_doc(pages=[p1, p2])

        p1_text_before = p1.text
        p2_text_before = p2.text
        doc_chars_before = doc.total_chars
        doc_page_count_before = doc.page_count

        chunk_document(doc, make_doc_metadata())

        self.assertEqual(p1.text, p1_text_before)
        self.assertEqual(p2.text, p2_text_before)
        self.assertEqual(doc.total_chars, doc_chars_before)
        self.assertEqual(doc.page_count, doc_page_count_before)

    def test_13_no_empty_substantive_chunks(self):
        """No emitted chunk contains empty or whitespace-only substantive body text."""
        p1 = PageText(
            page_number=1,
            text=(
                "\n\n\n"
                "1. First valid paragraph text.\n"
                "   \n"
                "2. Second valid paragraph text.\n"
            ),
            raw_text="...",
            char_count=60,
        )
        doc = make_parsed_doc(pages=[p1])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        for c in chunks:
            lines = c.text.split("\n", 1)
            body = lines[1].strip() if len(lines) > 1 else ""
            self.assertTrue(len(body) > 0, f"Chunk {c.chunk_id} has empty body")

    def test_14_all_final_chunks_within_size_policy_except_tables(self):
        """All non-table chunks are <= MAX_CHUNK_TOKENS."""
        s1 = "Sentence 1. " + " ".join(["word"] * 400) + "."
        s2 = "Sentence 2. " + " ".join(["word"] * 400) + "."
        s3 = "Sentence 3. " + " ".join(["word"] * 400) + "."
        p1 = PageText(page_number=1, text=f"1. {s1}\n{s2}\n{s3}\n", raw_text="...", char_count=3000)
        doc = make_parsed_doc(pages=[p1])
        meta = make_doc_metadata()
        chunks = chunk_document(doc, meta)
        for c in chunks:
            if not c.chunk_type.startswith("table"):
                self.assertLessEqual(c.token_count, MAX_CHUNK_TOKENS)

    def test_15_errors_on_invalid_input_or_missing_metadata(self):
        """Raises TypeError or ValueError predictably on invalid input."""
        meta = make_doc_metadata()

        # Non-ParsedDocument input
        with self.assertRaises(TypeError):
            chunk_document("not_a_parsed_doc", meta)  # type: ignore

        with self.assertRaises(TypeError):
            chunk_document(None, meta)  # type: ignore

        # Non-dict metadata input
        doc = make_parsed_doc(pages=[])
        with self.assertRaises(TypeError):
            chunk_document(doc, "not_a_dict")  # type: ignore

        # Missing required metadata keys
        with self.assertRaises(ValueError):
            chunk_document(doc, {"doc_id": "test"})  # missing source_filename, circular_number

        with self.assertRaises(ValueError):
            chunk_document(doc, {"doc_id": "test", "source_filename": "test.pdf"})  # missing circular_number


if __name__ == "__main__":
    unittest.main()
