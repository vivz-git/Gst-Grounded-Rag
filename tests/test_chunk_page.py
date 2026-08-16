"""
Unit tests for chunk_page() and ChunkerState in ingestion/chunker.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.pdf_parser import PageText
from ingestion.chunker import (
    ChunkerState,
    chunk_page,
    flush_chunker_state,
)


class TestChunkPage(unittest.TestCase):
    """Test suite for page-level section-aware chunking."""

    def setUp(self):
        self.doc_meta = {
            "doc_id": "Circular-172-04-2022-GST",
            "source_filename": "Circular-172-04-2022-GST.pdf",
            "circular_number": "172/04/2022-GST",
            "date_issued": "2022-07-06",
        }

    def test_1_simple_top_level_paragraph(self):
        """Test a single top-level paragraph on a page."""
        state = ChunkerState()
        page = PageText(
            page_number=1,
            text="1. In terms of Section 16(2) of the CGST Act, 2017, input tax credit reversal is clarified.",
            raw_text="...",
            char_count=90,
        )
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.section_path, "1")
        self.assertEqual(c.parent_section, "")
        self.assertEqual(c.chunk_type, "paragraph")
        self.assertEqual(c.page_start, 1)
        self.assertEqual(c.page_end, 1)
        self.assertIn("Section 16(2)", c.text)

    def test_2_multiple_paragraphs_on_one_page(self):
        """Test multiple numbered paragraphs on a single page."""
        state = ChunkerState()
        page_text = (
            "1. First paragraph regarding scope of credit.\n"
            "2. Second paragraph regarding formula under Rule 42.\n"
            "3. Third paragraph regarding reporting in GSTR-3B."
        )
        page = PageText(page_number=1, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([c.section_path for c in chunks], ["1", "2", "3"])
        self.assertEqual([c.chunk_type for c in chunks], ["paragraph", "paragraph", "paragraph"])

    def test_3_marker_on_one_line_body_on_next_line(self):
        """Test marker on standalone line with body text starting on following line."""
        state = ChunkerState()
        page_text = "3.\nFiling of refund application requires temporary registration on portal."
        page = PageText(page_number=2, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.section_path, "3")
        self.assertIn("Filing of refund application", c.text)

    def test_4_marker_and_body_on_same_line(self):
        """Test marker and body text occurring together on the same line."""
        state = ChunkerState()
        page_text = "4.1 Separate applications for refund have to be filed in respect of different suppliers."
        page = PageText(page_number=3, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.section_path, "4.1")
        self.assertEqual(c.parent_section, "4")
        self.assertEqual(c.chunk_type, "subparagraph")
        self.assertIn("Separate applications", c.text)

    def test_5_nested_clause_and_roman_structure(self):
        """Test deep hierarchical nesting with decimal, clause, and roman markers."""
        state = ChunkerState()
        page_text = (
            "4.1 Main heading for sub-section\n"
            "(a) First clause on eligible supplies\n"
            "(i) Condition for supplier certificate\n"
            "(ii) Condition for CA verification\n"
            "(b) Second clause on time limit"
        )
        page = PageText(page_number=4, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 5)
        paths = [c.section_path for c in chunks]
        parents = [c.parent_section for c in chunks]
        types = [c.chunk_type for c in chunks]

        self.assertEqual(paths, ["4.1", "4.1.(a)", "4.1.(a).(i)", "4.1.(a).(ii)", "4.1.(b)"])
        self.assertEqual(parents, ["4", "4.1", "4.1.(a)", "4.1.(a)", "4.1"])
        self.assertEqual(types, ["subparagraph", "clause", "roman_item", "roman_item", "clause"])

    def test_6_continuation_lines_staying_in_current_chunk(self):
        """Test that prose and proviso continuation lines do not spawn separate chunks."""
        state = ChunkerState()
        page_text = (
            "1. Subject matter of the circular.\n"
            "This continues on the next line with further statutory context.\n"
            "Provided that nothing contained in this clause shall apply to motor vehicles.\n"
            "Provided further that the said recipient has paid the supplier.\n"
            "2. Next section."
        )
        page = PageText(page_number=1, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 2)
        c1 = chunks[0]
        self.assertEqual(c1.section_path, "1")
        self.assertIn("Provided that", c1.text)
        self.assertIn("Provided further that", c1.text)

    def test_7_section_title_propagation(self):
        """Test that a section heading attaches as metadata and prefix header to subsequent chunks."""
        state = ChunkerState()
        page_text = (
            "Perquisites provided by employer to the employees as per contractual agreement\n"
            "5. Whether various perquisites are liable for GST?\n"
            "1. Services by employee to employer are covered under Schedule III."
        )
        page = PageText(page_number=4, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c.section_title, "Perquisites provided by employer to the employees as per contractual agreement")
            self.assertIn("Perquisites provided by employer", c.text.splitlines()[0])

    def test_8_cross_page_state_preservation(self):
        """Test that a paragraph spanning across two PageText objects preserves state and page spans."""
        state = ChunkerState()
        page1 = PageText(
            page_number=1,
            text="1. Paragraph starts on page 1 discussing ITC reversal under Rule 42",
            raw_text="...",
            char_count=70,
        )
        page2 = PageText(
            page_number=2,
            text=(
                "and continues onto page 2 explaining formula in detail.\n"
                "2. Second paragraph starting on page 2."
            ),
            raw_text="...",
            char_count=100,
        )

        # Page 1 does not flush
        c1 = chunk_page(page1, self.doc_meta, state, flush_at_end=False)
        self.assertEqual(len(c1), 0)  # Still buffering open paragraph

        # Page 2 finishes Para 1 and starts Para 2
        c2 = chunk_page(page2, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(c2), 2)

        para1_chunk = c2[0]
        self.assertEqual(para1_chunk.section_path, "1")
        self.assertEqual(para1_chunk.page_start, 1)
        self.assertEqual(para1_chunk.page_end, 2)
        self.assertIn("Paragraph starts on page 1", para1_chunk.text)
        self.assertIn("continues onto page 2", para1_chunk.text)

        para2_chunk = c2[1]
        self.assertEqual(para2_chunk.section_path, "2")
        self.assertEqual(para2_chunk.page_start, 2)
        self.assertEqual(para2_chunk.page_end, 2)

    def test_9_table_page_becomes_exactly_one_atomic_chunk(self):
        """Test that a page with has_table=True is emitted as a single atomic chunk."""
        state = ChunkerState()
        table_page = PageText(
            page_number=5,
            text="Whether amount in electronic credit ledger can be used? Output tax only.",
            raw_text="...",
            char_count=80,
            has_table=True,
        )
        chunks = chunk_page(table_page, self.doc_meta, state)
        self.assertEqual(len(chunks), 1)
        tc = chunks[0]
        self.assertEqual(tc.chunk_type, "table_qa")
        self.assertEqual(tc.page_start, 5)
        self.assertEqual(tc.page_end, 5)
        self.assertIn("Whether amount in electronic credit ledger", tc.text)

    def test_10_context_header_generation(self):
        """Test exact format of inherited context header."""
        state = ChunkerState()
        state.current_section_title = "Refund Procedure"
        page = PageText(
            page_number=3,
            text="4.2 Relevant date for refund application.",
            raw_text="...",
            char_count=40,
        )
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        self.assertEqual(len(chunks), 1)
        expected_header = "[172/04/2022-GST | 4.2 | Refund Procedure]"
        self.assertTrue(chunks[0].text.startswith(expected_header))

    def test_11_deterministic_chunk_id_generation(self):
        """Test that running the chunker twice produces identical deterministic chunk IDs."""
        page = PageText(
            page_number=1,
            text="1. First paragraph.\n2. Second paragraph.",
            raw_text="...",
            char_count=50,
        )
        state1 = ChunkerState()
        chunks1 = chunk_page(page, self.doc_meta, state1, flush_at_end=True)

        state2 = ChunkerState()
        chunks2 = chunk_page(page, self.doc_meta, state2, flush_at_end=True)

        self.assertEqual([c.chunk_id for c in chunks1], [c.chunk_id for c in chunks2])
        self.assertEqual(chunks1[0].chunk_id, "Circular-172-04-2022-GST__1__0001")
        self.assertEqual(chunks1[1].chunk_id, "Circular-172-04-2022-GST__2__0002")

    def test_12_no_empty_substantive_chunks(self):
        """Test that an isolated marker line does not produce an empty chunk."""
        state = ChunkerState()
        page_text = "1.\n\n2. Real substantive paragraph text."
        page = PageText(page_number=1, text=page_text, raw_text="...", char_count=len(page_text))
        chunks = chunk_page(page, self.doc_meta, state, flush_at_end=True)
        # Should only emit Para 2, ignoring isolated empty marker 1
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_path, "2")
        self.assertIn("Real substantive paragraph text", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
