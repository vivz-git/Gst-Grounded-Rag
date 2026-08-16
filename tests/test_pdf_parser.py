import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pymupdf as fitz

from ingestion.pdf_parser import (
    clean_page_text,
    normalize_unicode_artifacts,
    clean_line_whitespace,
    parse_pdf,
    ScannedPDFError,
    CBIC_RUNNING_HEADER_PATTERNS,
)


class TestPDFParserCleaning(unittest.TestCase):

    def test_unicode_normalization(self):
        """Test curly quotes and en/em dash normalization."""
        raw_text = 'Section 16 \u201cCGST Rules\u201d \u2018proviso\u2019 \u2013 rate \u2014 schedule'
        normalized = normalize_unicode_artifacts(raw_text)
        expected = 'Section 16 "CGST Rules" \'proviso\' - rate - schedule'
        self.assertEqual(normalized, expected)

    def test_circular_header_removed_on_continuation_pages(self):
        """Test repeated Circular No. running headers are stripped on pages 2+."""
        page2_text = (
            "Circular No. 170/02/2022-GST\n"
            "under section 10 of the CGST Act (composition taxable persons)\n"
            "and to UIN holders."
        )
        cleaned = clean_page_text(page2_text, is_first_page=False)
        self.assertNotIn("Circular No. 170/02/2022-GST", cleaned)
        self.assertIn("under section 10 of the CGST Act", cleaned)

        # Test spacing variants
        page2_spaced = (
            "Circular No.  183/15/2022 - GST\n"
            "ITC available as per FORM GSTR-2A."
        )
        cleaned_spaced = clean_page_text(page2_spaced, is_first_page=False)
        self.assertNotIn("Circular No.  183/15/2022 - GST", cleaned_spaced)
        self.assertIn("ITC available as per FORM GSTR-2A", cleaned_spaced)

    def test_circular_header_preserved_on_first_page(self):
        """Test Circular No. is kept intact on page 1."""
        page1_text = (
            "Circular No. 170/02/2022-GST\n"
            "F.No. CBIC-20001/2/2022-GST\n"
            "Government of India\n"
            "1. Subject: Clarification on ITC."
        )
        cleaned = clean_page_text(page1_text, is_first_page=True)
        self.assertIn("Circular No. 170/02/2022-GST", cleaned)
        self.assertIn("Government of India", cleaned)
        self.assertIn("1. Subject: Clarification on ITC.", cleaned)

    def test_mid_paragraph_circular_reference_not_removed(self):
        """Test that in-text mentions of circulars inside paragraphs are preserved."""
        text = "This clarification is issued vide Circular No. 170/02/2022-GST dated 6th July 2022."
        cleaned = clean_page_text(text, is_first_page=False)
        self.assertEqual(cleaned, text)

    def test_page_number_removal(self):
        """Test that standalone page number lines are removed."""
        text = (
            "Some legal text line.\n"
            "Page 2 of 7\n"
            "- 3 -\n"
            "Next legal text line."
        )
        cleaned = clean_page_text(text, is_first_page=False)
        self.assertNotIn("Page 2 of 7", cleaned)
        self.assertNotIn("- 3 -", cleaned)
        self.assertIn("Some legal text line.", cleaned)
        self.assertIn("Next legal text line.", cleaned)

    def test_indentation_and_paragraph_markers_preserved(self):
        """Test that markers like 1., 2., (a), (i) and leading indents are preserved."""
        text = (
            "1. Main paragraph heading\n"
            "  (a) First sub-clause regarding eligible credit\n"
            "    (i) Specific condition under rule 42\n"
            "2. Second paragraph heading"
        )
        cleaned = clean_page_text(text, is_first_page=False)
        lines = cleaned.splitlines()
        self.assertEqual(lines[0], "1. Main paragraph heading")
        self.assertEqual(lines[1], "  (a) First sub-clause regarding eligible credit")
        self.assertEqual(lines[2], "    (i) Specific condition under rule 42")
        self.assertEqual(lines[3], "2. Second paragraph heading")


class TestPDFParserIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("data/test_fixtures")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            self.test_dir.rmdir()
        except Exception:
            pass

    def test_multi_page_pdf_parsing_and_scanned_detection(self):
        """Test 2-page PDF where page 2 running header is stripped, and scanned PDF rejection."""
        # 1. Multi-page valid PDF
        pdf_path = self.test_dir / "multi_page_test.pdf"
        doc = fitz.open()
        
        # Page 1
        p1 = doc.new_page()
        p1.insert_text(
            (50, 50),
            "Circular No. 199/11/2023-GST\n"
            "Government of India\n"
            "Ministry of Finance\n"
            "1. Clarification regarding taxability of services.\n"
            "Page 1 of 2"
        )
        # Page 2
        p2 = doc.new_page()
        p2.insert_text(
            (50, 50),
            "Circular No. 199/11/2023-GST\n"
            "Government of India\n"
            "Page 2 of 2\n"
            "2. Input services distribution under Section 20."
        )
        doc.save(str(pdf_path))
        doc.close()

        parsed = parse_pdf(pdf_path)
        self.assertEqual(parsed.page_count, 2)
        
        # Page 1 must have circular header & government
        self.assertIn("Circular No. 199/11/2023-GST", parsed.pages[0].text)
        self.assertIn("Government of India", parsed.pages[0].text)
        self.assertNotIn("Page 1 of 2", parsed.pages[0].text)

        # Page 2 must have running headers stripped
        self.assertNotIn("Circular No. 199/11/2023-GST", parsed.pages[1].text)
        self.assertNotIn("Government of India", parsed.pages[1].text)
        self.assertNotIn("Page 2 of 2", parsed.pages[1].text)
        self.assertIn("2. Input services distribution under Section 20.", parsed.pages[1].text)

        # 2. Scanned PDF detection
        scanned_path = self.test_dir / "scanned_test.pdf"
        doc_s = fitz.open()
        doc_s.new_page()  # blank
        doc_s.new_page()  # blank
        doc_s.save(str(scanned_path))
        doc_s.close()

        with self.assertRaises(ScannedPDFError):
            parse_pdf(scanned_path)


if __name__ == "__main__":
    unittest.main()
