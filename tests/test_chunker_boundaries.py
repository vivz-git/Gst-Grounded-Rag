"""
Exhaustive unit tests for boundary detection in ingestion/chunker.py.
Tests all top-level, decimal, lettered, roman, bullet, heading, and non-boundary cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.chunker import (
    Chunk,
    detect_boundary,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
)


class TestChunkDataclass(unittest.TestCase):
    """Test Chunk dataclass instantiation and default values."""

    def test_chunk_instantiation(self):
        chunk = Chunk(
            chunk_id="circular_172__5__0001",
            doc_id="circular_172",
            source_filename="Circular-172-04-2022-GST.pdf",
            page_start=4,
            page_end=4,
            section_path="5",
            parent_section="",
            section_title="Perquisites provided by employer",
            text="[172/04/2022-GST | Para 5] Services by employee are exempt...",
            token_count=12,
            chunk_type="paragraph",
            parsing_method="structural",
            circular_number="172/04/2022-GST",
            date_issued="2022-07-06",
        )
        self.assertEqual(chunk.chunk_id, "circular_172__5__0001")
        self.assertEqual(chunk.page_start, 4)
        self.assertEqual(chunk.chunk_type, "paragraph")
        self.assertEqual(chunk.parsing_method, "structural")
        self.assertIsInstance(chunk.metadata, dict)

    def test_settings_imports(self):
        """Ensure settings constraints are imported properly from config.settings."""
        self.assertEqual(MAX_CHUNK_TOKENS, 1000)
        self.assertEqual(MIN_CHUNK_TOKENS, 30)


class TestTopLevelBoundaries(unittest.TestCase):
    """Test top-level numbered paragraph detection."""

    def test_standalone_top_level_numbers(self):
        self.assertEqual(detect_boundary("1."), ("top_level", "1"))
        self.assertEqual(detect_boundary("2."), ("top_level", "2"))
        self.assertEqual(detect_boundary("12."), ("top_level", "12"))
        self.assertEqual(detect_boundary("  3.  "), ("top_level", "3"))

    def test_top_level_with_trailing_text(self):
        self.assertEqual(
            detect_boundary("3. Filing of refund application"),
            ("top_level", "3"),
        )
        self.assertEqual(
            detect_boundary("1. In terms of Section 16(2) of the CGST Act,"),
            ("top_level", "1"),
        )


class TestDecimalBoundaries(unittest.TestCase):
    """Test decimal sub-paragraph detection."""

    def test_standalone_decimal_subparas(self):
        self.assertEqual(detect_boundary("1.1"), ("decimal", "1.1"))
        self.assertEqual(detect_boundary("4.2"), ("decimal", "4.2"))
        self.assertEqual(detect_boundary("4.1.2"), ("decimal", "4.1.2"))
        self.assertEqual(detect_boundary("  2.1  "), ("decimal", "2.1"))

    def test_decimal_with_trailing_text_and_optional_dot(self):
        self.assertEqual(
            detect_boundary("2.1 In order to enable such unregistered person to file application"),
            ("decimal", "2.1"),
        )
        self.assertEqual(
            detect_boundary("4.1.2 In cases, where difference between the ITC claimed"),
            ("decimal", "4.1.2"),
        )
        self.assertEqual(
            detect_boundary("1.1.2. something"),
            ("decimal", "1.1.2"),
        )


class TestLetteredClauseBoundaries(unittest.TestCase):
    """Test lettered clause detection (a), (b), etc."""

    def test_lettered_clauses(self):
        self.assertEqual(detect_boundary("(a) text"), ("clause", "(a)"))
        self.assertEqual(detect_boundary("(b) text"), ("clause", "(b)"))
        self.assertEqual(detect_boundary("  (c) eligible input tax credit"), ("clause", "(c)"))
        self.assertEqual(detect_boundary("(z) final sub-clause"), ("clause", "(z)"))
        self.assertEqual(detect_boundary("(a)"), ("clause", "(a)"))


class TestRomanBoundaries(unittest.TestCase):
    """Test roman numeral detection (i), (ii), i., ii."""

    def test_parenthesized_roman_numerals(self):
        self.assertEqual(detect_boundary("(i) text"), ("roman", "(i)"))
        self.assertEqual(detect_boundary("(ii) text"), ("roman", "(ii)"))
        self.assertEqual(detect_boundary("(iii) text"), ("roman", "(iii)"))
        self.assertEqual(detect_boundary("(iv) text"), ("roman", "(iv)"))
        self.assertEqual(detect_boundary("(v) text"), ("roman", "(v)"))
        self.assertEqual(detect_boundary("  (vi) text"), ("roman", "(vi)"))
        self.assertEqual(detect_boundary("(x) text"), ("roman", "(x)"))

    def test_roman_vs_clause_priority(self):
        # (i) and (v) must be recognized as roman, not lettered clause
        self.assertEqual(detect_boundary("(i) text")[0], "roman")
        self.assertEqual(detect_boundary("(v) text")[0], "roman")
        self.assertEqual(detect_boundary("(a) text")[0], "clause")
        self.assertEqual(detect_boundary("(b) text")[0], "clause")

    def test_bare_roman_with_enumeration_context(self):
        # Bare roman with text when in enumeration context
        self.assertEqual(
            detect_boundary("i. refund claimed by the recipients", in_enumeration=True),
            ("roman", "i."),
        )
        self.assertEqual(
            detect_boundary("ii. interpretation of section 17(5)", in_enumeration=True),
            ("roman", "ii."),
        )
        self.assertEqual(
            detect_boundary("iii. perquisites provided by employer", in_enumeration=True),
            ("roman", "iii."),
        )
        self.assertEqual(
            detect_boundary("iv. utilisation of the amounts", in_enumeration=True),
            ("roman", "iv."),
        )

    def test_bare_roman_standalone_line(self):
        # Standalone "i." line as seen in Circular 172 page 1
        self.assertEqual(detect_boundary("i."), ("roman", "i."))
        self.assertEqual(detect_boundary("  ii.  "), ("roman", "ii."))
        self.assertEqual(detect_boundary("iii."), ("roman", "iii."))
        self.assertEqual(detect_boundary("iv."), ("roman", "iv."))

    def test_bare_roman_without_context_avoids_prose_collision(self):
        # Without enumeration context or allow flag, bare roman in prose is rejected
        self.assertIsNone(detect_boundary("i. refund claimed by the recipients", in_enumeration=False, allow_bare_roman=False))


class TestBulletBoundaries(unittest.TestCase):
    """Test bullet and dash-prefixed item detection."""

    def test_bullets_and_dashes(self):
        self.assertEqual(detect_boundary("- text"), ("bullet", "-"))
        self.assertEqual(detect_boundary("– text"), ("bullet", "–"))  # en-dash
        self.assertEqual(detect_boundary("— text"), ("bullet", "—"))  # em-dash
        self.assertEqual(detect_boundary("• text"), ("bullet", "•"))
        self.assertEqual(detect_boundary("* text"), ("bullet", "*"))


class TestSectionHeadingBoundaries(unittest.TestCase):
    """Test conservative section heading detection from real GST circular patterns."""

    def test_valid_corpus_headings(self):
        self.assertEqual(
            detect_boundary("Perquisites provided by employer to the employees as per contractual agreement"),
            ("heading", "Perquisites provided by employer to the employees as per contractual agreement"),
        )
        self.assertEqual(
            detect_boundary("Filing of refund application"),
            ("heading", "Filing of refund application"),
        )
        self.assertEqual(
            detect_boundary("Relevant date for filing of refund:"),
            ("heading", "Relevant date for filing of refund"),
        )
        self.assertEqual(
            detect_boundary("Utilisation of the amounts available in the electronic credit ledger and cash ledger"),
            ("heading", "Utilisation of the amounts available in the electronic credit ledger and cash ledger"),
        )


class TestNonBoundaries(unittest.TestCase):
    """Test that ordinary prose, statutory references, provisos, dates, and lowercase text are NOT boundaries."""

    def test_provisos_and_statutory_references(self):
        self.assertIsNone(detect_boundary("Provided that nothing contained in this clause shall apply..."))
        self.assertIsNone(detect_boundary("Provided further that the said recipient has paid..."))
        self.assertIsNone(detect_boundary("Section 17(5) applies to input tax credit restrictions."))
        self.assertIsNone(detect_boundary("Rule 42 and Rule 43 of the CGST Rules provide the formula."))
        self.assertIsNone(detect_boundary("Table 3.2 of FORM GSTR-3B requires declaration."))
        self.assertIsNone(detect_boundary("Schedule III to the CGST Act provides that services by employee..."))

    def test_dates_and_citations(self):
        self.assertIsNone(detect_boundary("12.03.2021 only for enabling them to claim such refunds."))
        self.assertIsNone(detect_boundary("26.12.2022 to provide for the documents required to be furnished."))
        self.assertIsNone(detect_boundary("1st July, 2017 itself. In view of this, various representations..."))

    def test_ordinary_prose_and_boilerplate(self):
        self.assertIsNone(detect_boundary("The proper officer shall first seek the details from the registered person."))
        self.assertIsNone(detect_boundary("Government of India"))
        self.assertIsNone(detect_boundary("Ministry of Finance"))
        self.assertIsNone(detect_boundary("GST Policy Wing"))
        self.assertIsNone(detect_boundary("Subject: Clarification on various issue pertaining to GST- reg."))
        self.assertIsNone(detect_boundary("Difficulty, if any, in the implementation of this Circular..."))
        self.assertIsNone(detect_boundary("Hindi version would follow."))

    def test_lowercase_continuation_lines(self):
        self.assertIsNone(detect_boundary("under section 10 of the CGST Act (composition taxable persons)"))
        self.assertIsNone(detect_boundary("and to UIN holders, as required to be declared in Table 3.2"))
        self.assertIsNone(detect_boundary("category of goods or services or both"))

    def test_quoted_references_and_urls(self):
        self.assertIsNone(detect_boundary('"CGST Rules" only with effect from 9th October 2019.'))
        self.assertIsNone(detect_boundary('https://udin.icai.org/search-udin'))

    def test_empty_and_whitespace(self):
        self.assertIsNone(detect_boundary(""))
        self.assertIsNone(detect_boundary("   "))
        self.assertIsNone(detect_boundary("\n\t"))


if __name__ == "__main__":
    unittest.main()
