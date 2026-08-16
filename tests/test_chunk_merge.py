"""
Unit tests for merge_small_fragments() in ingestion/chunker.py.
Covers upward merge, downward merge, compatibility guards, table preservation,
MAX_CHUNK_TOKENS budget checks, page span inheritance, and deterministic ordering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.chunker import (
    Chunk,
    merge_small_fragments,
    _extract_substantive_text,
    _are_chunks_compatible,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
)


def make_chunk(
    chunk_id: str = "doc__1__0001",
    doc_id: str = "doc",
    source_filename: str = "doc.pdf",
    page_start: int = 1,
    page_end: int = 1,
    section_path: str = "1",
    parent_section: str = "",
    section_title: str = "",
    body_text: str = "This is a standard body text for chunk.",
    chunk_type: str = "paragraph",
    circular_number: str = "172/04/2022-GST",
    date_issued: str = "2022-07-06",
) -> Chunk:
    """Helper to construct test Chunk with properly formatted context header."""
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


class TestMergeSmallFragments(unittest.TestCase):
    """Test suite for small fragment merging."""

    def test_1_small_chunk_merges_upward(self):
        """A small chunk (< MIN_CHUNK_TOKENS) merges into preceding compatible chunk."""
        c0 = make_chunk(
            chunk_id="doc__1__0001",
            section_path="1",
            body_text=" ".join(["word"] * 40),  # 40 words (> 30 tokens)
        )
        c1 = make_chunk(
            chunk_id="doc__1__0002",
            section_path="1",
            body_text="short fragment.",  # ~7 tokens (< 30 tokens)
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].section_path, "1")
        self.assertIn("short fragment.", merged[0].text)
        self.assertGreater(merged[0].token_count, c0.token_count)

    def test_2_small_chunk_merges_downward_when_first(self):
        """A small chunk at the start of a document merges downward into following compatible chunk."""
        c0 = make_chunk(
            chunk_id="doc__3__0001",
            section_path="3",
            body_text="3. Intro marker.",  # ~8 tokens (< 30)
        )
        c1 = make_chunk(
            chunk_id="doc__3_1__0002",
            section_path="3.1",
            parent_section="3",
            body_text=" ".join(["subparagraph text"] * 15),  # ~35 tokens
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].section_path, "3.1")
        # Substantive text of c0 must precede c1
        substantive = _extract_substantive_text(merged[0].text)
        self.assertTrue(substantive.startswith("3. Intro marker."))

    def test_3_compatible_section_fragment_merges_correctly(self):
        """Two adjacent chunks with identical section_path merge into one coherent chunk."""
        c0 = make_chunk(
            chunk_id="doc__4_1__0001",
            section_path="4.1",
            parent_section="4",
            body_text="4.1 First part of supplier verification condition.",
        )
        c1 = make_chunk(
            chunk_id="doc__4_1__0002",
            section_path="4.1",
            parent_section="4",
            body_text="Second part of condition.",
        )
        # Force c1 to be small
        self.assertLess(c1.token_count, MIN_CHUNK_TOKENS)
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].section_path, "4.1")
        self.assertIn("First part", merged[0].text)
        self.assertIn("Second part", merged[0].text)

    def test_4_incompatible_chunk_types_do_not_merge(self):
        """Unrelated top-level legal sections do not merge even if one is small."""
        c0 = make_chunk(
            chunk_id="doc__5__0001",
            section_path="5",
            parent_section="",
            body_text="5. Short para.",  # < 30 tokens
        )
        c1 = make_chunk(
            chunk_id="doc__6__0002",
            section_path="6",
            parent_section="",
            body_text=" ".join(["Section 6 ongoing proceedings body."] * 10),
        )
        # Unrelated sections '5' and '6' must NOT merge
        self.assertFalse(_are_chunks_compatible(c0, c1))
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].section_path, "5")
        self.assertEqual(merged[1].section_path, "6")

    def test_5_table_chunks_are_not_accidentally_merged(self):
        """Table chunks are atomic and must never merge with normal chunks."""
        c0 = make_chunk(
            chunk_id="doc__2__0001",
            section_path="2",
            body_text="Small para before table.",  # < 30 tokens
        )
        c1 = make_chunk(
            chunk_id="doc__p_2__0002",
            section_path="p.2",
            chunk_type="table_raw",
            body_text=" ".join(["Table data row content"] * 20),
        )
        self.assertFalse(_are_chunks_compatible(c0, c1))
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1].chunk_type, "table_raw")

    def test_6_max_chunk_tokens_prevents_unsafe_merging(self):
        """A merge is aborted if the resulting chunk would exceed MAX_CHUNK_TOKENS."""
        large_body = " ".join(["word"] * (MAX_CHUNK_TOKENS - 10))
        c0 = make_chunk(
            chunk_id="doc__1__0001",
            section_path="1",
            body_text=large_body,
        )
        c1 = make_chunk(
            chunk_id="doc__1__0002",
            section_path="1",
            body_text=" ".join(["extra"] * 20),  # < 30 tokens
        )
        # Merging would exceed MAX_CHUNK_TOKENS (1000)
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 2)  # Kept separate to prevent overflow

    def test_7_page_start_and_page_end_preserved_correctly(self):
        """Page span becomes min(start pages) and max(end pages)."""
        c0 = make_chunk(
            chunk_id="doc__1__0001",
            page_start=1,
            page_end=2,
            section_path="1",
            body_text=" ".join(["word"] * 40),
        )
        c1 = make_chunk(
            chunk_id="doc__1__0002",
            page_start=2,
            page_end=3,
            section_path="1",
            body_text="Small continuation.",
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].page_start, 1)
        self.assertEqual(merged[0].page_end, 3)

    def test_8_token_count_recomputed_correctly(self):
        """Token count matches the exact whitespace token count of the new merged text."""
        c0 = make_chunk(
            chunk_id="doc__1__0001",
            section_path="1",
            body_text="One two three four five six seven eight nine ten.",
        )
        c1 = make_chunk(
            chunk_id="doc__1__0002",
            section_path="1",
            body_text="Alpha beta gamma delta epsilon.",
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        expected_tokens = len(merged[0].text.split())
        self.assertEqual(merged[0].token_count, expected_tokens)

    def test_9_section_path_and_parent_metadata_remain_coherent(self):
        """Target chunk's section_path and parent metadata are retained."""
        c0 = make_chunk(
            chunk_id="doc__4_1__0001",
            section_path="4.1",
            parent_section="4",
            section_title="Refund Verification",
            body_text=" ".join(["verification rules"] * 20),
        )
        c1 = make_chunk(
            chunk_id="doc__4_1_a__0002",
            section_path="4.1.(a)",
            parent_section="4.1",
            section_title="Refund Verification",
            body_text="(a) CA certificate.",  # < 30 tokens
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].section_path, "4.1")
        self.assertEqual(merged[0].parent_section, "4")
        self.assertEqual(merged[0].section_title, "Refund Verification")

    def test_10_already_valid_chunks_remain_unchanged(self):
        """Chunks >= MIN_CHUNK_TOKENS are not merged."""
        c0 = make_chunk(
            chunk_id="doc__1__0001",
            section_path="1",
            body_text=" ".join(["valid large paragraph text"] * 10),  # ~40 tokens
        )
        c1 = make_chunk(
            chunk_id="doc__2__0002",
            section_path="2",
            body_text=" ".join(["another valid paragraph text"] * 10),  # ~40 tokens
        )
        merged = merge_small_fragments([c0, c1])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].section_path, "1")
        self.assertEqual(merged[1].section_path, "2")

    def test_11_multiple_consecutive_small_fragments_handled_deterministically(self):
        """Multiple consecutive small fragments under the same section merge cleanly."""
        c0 = make_chunk(
            chunk_id="doc__preamble__0001",
            section_path="preamble",
            chunk_type="preamble",
            body_text="Circular No. 183/15/2022-GST",  # ~8 tokens
        )
        c1 = make_chunk(
            chunk_id="doc__preamble__0002",
            section_path="preamble",
            chunk_type="preamble",
            body_text="F. No. CBIC-20001/2/2022-GST",  # ~8 tokens
        )
        c2 = make_chunk(
            chunk_id="doc__preamble__0003",
            section_path="preamble",
            chunk_type="preamble",
            body_text="Government of India Ministry of Finance Department of Revenue",  # ~12 tokens
        )
        merged = merge_small_fragments([c0, c1, c2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].section_path, "preamble")
        self.assertIn("183/15/2022-GST", merged[0].text)
        self.assertIn("Department of Revenue", merged[0].text)

    def test_12_input_order_remains_stable(self):
        """Chunks maintain their strict sequential order and deterministic IDs."""
        c0 = make_chunk(chunk_id="doc__1__0001", section_path="1", body_text=" ".join(["p1"] * 35))
        c1 = make_chunk(chunk_id="doc__2__0002", section_path="2", body_text=" ".join(["p2"] * 35))
        c2 = make_chunk(chunk_id="doc__3__0003", section_path="3", body_text=" ".join(["p3"] * 35))
        merged = merge_small_fragments([c0, c1, c2])
        self.assertEqual([c.chunk_id for c in merged], ["doc__1__0001", "doc__2__0002", "doc__3__0003"])
        self.assertEqual([c.section_path for c in merged], ["1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
