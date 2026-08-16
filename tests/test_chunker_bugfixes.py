"""
Regression tests for Bug 1 (false heading detection) and Bug 2 (cross-page
hyphenated-number misread as a new paragraph marker) in ingestion/chunker.py.
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
    detect_boundary,
    _is_conservative_section_heading,
)


# =============================================================================
# BUG 1: False Section Heading Detection
# =============================================================================

class TestHeadingDetectionRegressions(unittest.TestCase):
    """Regression tests ensuring the tightened heading detector is corpus-accurate."""

    # --- Real legitimate headings that MUST still be detected ---
    def test_real_heading_refund_claimed(self):
        result = detect_boundary("Refund claimed by the recipients of supplies regarded as deemed export")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "heading")

    def test_real_heading_perquisites(self):
        result = detect_boundary("Perquisites provided by employer to the employees as per contractual agreement")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "heading")

    def test_real_heading_filing_refund(self):
        result = detect_boundary("Filing of refund application")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "heading")

    def test_real_heading_relevant_date(self):
        result = detect_boundary("Relevant date for filing of refund")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "heading")

    def test_real_heading_clarification_section_17(self):
        result = detect_boundary("Clarification on various issues of section 17(5) of the CGST Act")
        # This ends with "Act" which is not a continuation word - should be heading
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "heading")

    # --- False positives from the real corpus that must NOT be detected as headings ---

    def test_false_positive_commissioners(self):
        # "Commissioners of Central Tax (All)" contains "(All)" -> not a heading
        result = detect_boundary("Commissioners of Central Tax (All)")
        self.assertIsNone(result)

    def test_false_positive_september_wrapped(self):
        # Long sentence fragment that was previously a false heading
        line = "September, 2018 till the due date of furnishing return for March, 2019, if supplier had not"
        self.assertFalse(_is_conservative_section_heading(line))
        self.assertIsNone(detect_boundary(line))

    def test_false_positive_cgst_rules_wrapped(self):
        line = "CGST Rules or for any other reasons are required to be made by the registered person, on his"
        self.assertFalse(_is_conservative_section_heading(line))
        self.assertIsNone(detect_boundary(line))

    def test_false_positive_in_this_context(self):
        line = "In this context, it is pertinent to mention that the facility of static month-wise auto-"
        self.assertFalse(_is_conservative_section_heading(line))

    def test_false_positive_ends_with_hyphen(self):
        # Any line ending with '-' is a wrapped word, not a heading
        self.assertFalse(_is_conservative_section_heading("Sub-clause (i) of clause (b) of sub-"))
        self.assertFalse(_is_conservative_section_heading("due date of furnishing return for March, 2018-"))

    def test_false_positive_ends_with_continuation_word_the(self):
        self.assertFalse(_is_conservative_section_heading(
            "For ease of understanding, the manner of reversals is being elucidated in the"
        ))

    def test_false_positive_ends_with_continuation_word_in(self):
        self.assertFalse(_is_conservative_section_heading(
            "A. Total ITC (eligible as well as ineligible) is being auto-populated from statement in"
        ))

    def test_false_positive_ends_with_continuation_word_and(self):
        self.assertFalse(_is_conservative_section_heading(
            "B. Registered person will report reversal of ITC, which are absolute in nature and are"
        ))

    def test_false_positive_ends_with_comma(self):
        self.assertFalse(_is_conservative_section_heading(
            "State / UT than that of place of supply. It is pertinent to mention that the ineligible ITC,"
        ))

    def test_false_positive_mid_sentence_period(self):
        # Contains '. [Uppercase]' -> is two sentences, not a heading
        self.assertFalse(_is_conservative_section_heading(
            "August, 2020. The statement provides invoice-wise total details of ITC available to the"
        ))

    def test_false_positive_principal_commissioner(self):
        # 'Principal Commissioner (GST)' - short identifier, not a heading
        self.assertFalse(_is_conservative_section_heading("Principal Commissioner (GST)"))

    def test_false_positive_person_name(self):
        self.assertFalse(_is_conservative_section_heading("Sanjay Mangal"))

    def test_false_positive_month_start(self):
        self.assertFalse(_is_conservative_section_heading(
            "September, 2018 till the due date of furnishing return"
        ))

    def test_false_positive_slash_in_line(self):
        # Lines with slashes are addressee fragments like "Madam/Sir" or "he/she"
        self.assertFalse(_is_conservative_section_heading(
            "Commissioners of Central Tax (All)/ Director General"
        ))

    def test_false_positive_ends_with_continuation_word_not(self):
        self.assertFalse(_is_conservative_section_heading(
            "Act provides that ITC shall not be"
        ))

    def test_false_positive_difficulty_if_any(self):
        # Already covered by PROSE_HEADING_EXCLUSIONS, but verify here too
        self.assertFalse(_is_conservative_section_heading(
            "Difficulty if any, in the implementation of this Circular may be brought to the"
        ))

    def test_false_positive_too_few_words(self):
        # Less than 3 alphabetic words
        self.assertFalse(_is_conservative_section_heading("Short line"))

    def test_ordinary_lowercase_prose(self):
        self.assertIsNone(detect_boundary("under section 17(5) of the CGST Act"))
        self.assertIsNone(detect_boundary("and the tax on such supplies has been paid"))

    def test_date_line_not_heading(self):
        self.assertIsNone(detect_boundary("New Delhi, Dated the 27th December, 2022"))


# =============================================================================
# BUG 2: Cross-Page Hyphenated Token Misread as New Paragraph
# =============================================================================

class TestCrossPageHyphenContinuation(unittest.TestCase):
    """Regression tests for cross-page hyphenated token continuation behavior."""

    def _doc_meta(self):
        return {
            "doc_id": "circular-183",
            "source_filename": "circular-183.pdf",
            "circular_number": "183/15/2022-GST",
            "date_issued": "",
        }

    def test_hyphen_year_split_is_not_new_paragraph(self):
        """
        The real corpus bug: page 4 ends with '2018-' and page 5 starts with
        '19. Further, these guidelines...' — '19.' must NOT be treated as Para 19.
        """
        state = ChunkerState()
        doc_meta = self._doc_meta()

        # Page 4: ends with a line containing a trailing hyphen
        page4 = PageText(
            page_number=4,
            text=(
                "5. These instructions apply for FY 2017-18 and 2018-\n"
            ),
            raw_text="...",
            char_count=60,
        )
        # Page 5: starts with what looks like "19." but is really "19. Further..."
        # continuing the hyphenated year "2018-19"
        page5 = PageText(
            page_number=5,
            text=(
                "19. Further, these guidelines are clarificatory in nature and may be applied.\n"
                "7. Difficulty, if any, in the implementation."
            ),
            raw_text="...",
            char_count=120,
        )

        c4 = chunk_page(page4, doc_meta, state, flush_at_end=False)
        c5 = chunk_page(page5, doc_meta, state, flush_at_end=True)

        all_chunks = c4 + c5
        all_paths = [c.section_path for c in all_chunks]

        # Must NOT have section_path == "19"
        self.assertNotIn("19", all_paths, "Section path '19' should not exist — it was a cross-page hyphen artifact")
        # Must have section_path "5" (the real paragraph opened on page 4)
        self.assertIn("5", all_paths)
        # Must have section_path "7" (the closing paragraph)
        self.assertIn("7", all_paths)

    def test_genuine_paragraph_19_at_page_boundary(self):
        """
        A genuine paragraph 19 appearing at the start of a page (when the previous
        page did NOT end with a hyphen) should still be detected as top_level.
        """
        state = ChunkerState()
        doc_meta = self._doc_meta()

        # Page 3: ends cleanly, no trailing hyphen
        page3 = PageText(
            page_number=3,
            text=(
                "18. The earlier provisions were clarified vide circular dated 12.03.2021.\n"
            ),
            raw_text="...",
            char_count=75,
        )
        # Page 4: starts with genuine paragraph 19
        page4 = PageText(
            page_number=4,
            text=(
                "19. Further instructions are as follows for clarity on assessment.\n"
            ),
            raw_text="...",
            char_count=70,
        )

        c3 = chunk_page(page3, doc_meta, state, flush_at_end=False)
        c4 = chunk_page(page4, doc_meta, state, flush_at_end=True)
        all_chunks = c3 + c4
        all_paths = [c.section_path for c in all_chunks]

        # Genuine "19." at page start with no prior hyphen must be detected
        self.assertIn("19", all_paths)

    def test_ordinary_hyphenated_word_continuation(self):
        """
        A normal hyphenated word wrap such as 'regis-' + 'tered person' should also
        be handled as a continuation, and 'tered' should not trigger any boundary.
        """
        state = ChunkerState()
        doc_meta = self._doc_meta()

        page1 = PageText(
            page_number=1,
            text="1. The registered person must comply with the require-\n",
            raw_text="...",
            char_count=55,
        )
        page2 = PageText(
            page_number=2,
            text="ments of Rule 42 and Rule 43 of the CGST Rules.\n2. Second paragraph here.\n",
            raw_text="...",
            char_count=75,
        )

        c1 = chunk_page(page1, doc_meta, state, flush_at_end=False)
        c2 = chunk_page(page2, doc_meta, state, flush_at_end=True)
        all_chunks = c1 + c2
        all_paths = [c.section_path for c in all_chunks]

        # "ments of Rule..." should be continuation of Para 1
        self.assertIn("1", all_paths)
        # Para 2 should be separate
        self.assertIn("2", all_paths)

    def test_trailing_hyphen_flag_cleared_after_use(self):
        """After consuming one continuation line, the trailing_hyphen_active flag resets."""
        state = ChunkerState()
        doc_meta = self._doc_meta()

        page1 = PageText(
            page_number=1,
            text="1. Paragraph one ending with hyphen-\n",
            raw_text="...",
            char_count=40,
        )
        page2 = PageText(
            page_number=2,
            text="continuation text.\n2. Real para two.\n",
            raw_text="...",
            char_count=40,
        )
        page3 = PageText(
            page_number=3,
            text="3. Para three standalone.\n",
            raw_text="...",
            char_count=30,
        )

        chunk_page(page1, doc_meta, state, flush_at_end=False)
        self.assertTrue(state.trailing_hyphen_active)
        chunk_page(page2, doc_meta, state, flush_at_end=False)
        # After page2 consumption, flag must be cleared (page2 doesn't end with hyphen)
        self.assertFalse(state.trailing_hyphen_active)
        chunk_page(page3, doc_meta, state, flush_at_end=True)


if __name__ == "__main__":
    unittest.main()
