"""
PDF Parser module for GST Circular RAG Assistant.

Extracts text from native-text PDFs using PyMuPDF (fitz), enforces native-text
quality gates, normalizes whitespace and boilerplate, and generates structured
ParsedDocument representations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF

from config.settings import PDF_DIR

# Set up module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ScannedPDFError(ValueError):
    """Raised when a PDF fails the native-text threshold and is identified as scanned."""
    pass


class EmptyPDFError(ValueError):
    """Raised when a PDF file is empty or contains zero pages."""
    pass


@dataclass
class PageText:
    """Represents extracted text and metadata for a single PDF page."""
    page_number: int  # 1-indexed
    text: str  # Cleaned text
    raw_text: str  # Original uncleaned text from PyMuPDF
    char_count: int
    has_table: bool = False


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    file_path: str
    file_name: str
    page_count: int
    pages: List[PageText]
    raw_text: str
    total_chars: int
    warnings: List[str] = field(default_factory=list)
    tables_detected: int = 0


# ==============================================================================
# Text Cleaning & Normalization Patterns
# ==============================================================================

# Unicode quotation and dash variant normalization mapping
UNICODE_REPLACEMENTS = {
    "\u201c": '"',  # Left double quotation mark
    "\u201d": '"',  # Right double quotation mark
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
}

# Regex for common page numbering footers/headers: "Page 1 of 5", "- 2 -", "Page 3", etc.
PAGE_NUMBER_REGEX = re.compile(
    r"^\s*(?:page\s*-?\s*\d+\s*(?:of\s*\d+)?|-?\s*\d+\s*-?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Regex to remove multiple horizontal spaces/tabs
HORIZONTAL_WHITESPACE_REGEX = re.compile(r"[ \t]+")

# Regex to normalize excessive linebreaks (3 or more -> 2)
EXCESSIVE_NEWLINES_REGEX = re.compile(r"\n{3,}")

# Common CBIC repeated header lines that may appear as running headers
CBIC_RUNNING_HEADER_PATTERNS = [
    re.compile(
        r"^\s*Circular\s+No\.?\s*\d+\s*(?:/\s*\d+\s*)*\s*-\s*GST\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*Government\s+of\s+India\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Ministry\s+of\s+Finance\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Department\s+of\s+Revenue\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"^\s*Central\s+Board\s+of\s+Indirect\s+Taxes\s+(?:and|&)\s+Customs\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*GST\s+Policy\s+Wing\s*$", re.IGNORECASE | re.MULTILINE),
]


def normalize_unicode_artifacts(text: str) -> str:
    """Normalize common Unicode quotation and dash artifacts from PDF extraction."""
    for char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def clean_line_whitespace(line: str) -> str:
    """
    Clean horizontal whitespace: preserve leading indentation while collapsing
    multiple internal spaces and removing trailing whitespace.
    """
    match = re.match(r"^([ \t]*)", line)
    indent = match.group(1) if match else ""
    content = line[len(indent):].rstrip()
    if not content:
        return ""
    indent_cleaned = indent.replace("\t", "  ")
    content_cleaned = HORIZONTAL_WHITESPACE_REGEX.sub(" ", content)
    return indent_cleaned + content_cleaned


def clean_page_text(text: str, is_first_page: bool = False) -> str:
    """
    Clean and normalize raw extracted page text.

    Args:
        text: Raw text from PyMuPDF page.
        is_first_page: If False, strips repeated running header boilerplate.

    Returns:
        Cleaned and normalized text string.
    """
    if not text:
        return ""

    # Normalize Unicode artifacts (curly quotes, en/em dashes)
    cleaned = normalize_unicode_artifacts(text)

    # Remove standalone page number lines
    cleaned = PAGE_NUMBER_REGEX.sub("", cleaned)

    # For continuation pages (> page 1), remove repeated standard running headers
    if not is_first_page:
        for pattern in CBIC_RUNNING_HEADER_PATTERNS:
            cleaned = pattern.sub("", cleaned)

    # Clean horizontal whitespace per line while preserving leading indentation
    lines = [clean_line_whitespace(line) for line in cleaned.splitlines()]

    # Reassemble text preserving paragraph linebreaks
    cleaned = "\n".join(lines)
    cleaned = EXCESSIVE_NEWLINES_REGEX.sub("\n\n", cleaned)

    return cleaned.strip()


def detect_tables_on_page(page: fitz.Page) -> int:
    """
    Detect tables on a page using PyMuPDF table finder if available.
    Tags detected tables for v2 processing without altering text.

    Args:
        page: PyMuPDF Page object.

    Returns:
        Number of tables detected on the page.
    """
    try:
        if hasattr(page, "find_tables"):
            tabs = page.find_tables()
            return len(tabs.tables) if tabs else 0
    except Exception as e:
        logger.debug(f"Table detection check skipped on page: {e}")
    return 0


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Parse a single PDF file into a ParsedDocument dataclass.

    Enforces native-text validation: rejects scanned documents where >50% of pages
    contain fewer than 50 characters of extractable text.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        ParsedDocument instance containing page texts and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyPDFError: If the PDF is empty or has 0 pages.
        ScannedPDFError: If the PDF fails the native-text quality check.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{path.name}': {e}") from e

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise EmptyPDFError(f"PDF '{path.name}' has 0 pages.")

    # --------------------------------------------------------------------------
    # Native-Text Quality Gate (Reject Scanned Documents)
    # --------------------------------------------------------------------------
    low_char_page_count = 0
    raw_pages_text: List[str] = []

    for i in range(page_count):
        page = doc[i]
        text = page.get_text("text") or ""
        raw_pages_text.append(text)
        if len(text.strip()) < 50:
            low_char_page_count += 1

    # If >50% of pages have <50 characters, flag as scanned PDF
    if (low_char_page_count / page_count) > 0.5:
        doc.close()
        raise ScannedPDFError(
            f"Scanned PDF detected — excluded from v1 corpus: '{path.name}' "
            f"({low_char_page_count}/{page_count} pages have <50 characters)"
        )

    # --------------------------------------------------------------------------
    # Extraction, Cleaning & Warning Logging
    # --------------------------------------------------------------------------
    pages: List[PageText] = []
    warnings: List[str] = []
    total_tables = 0

    for i in range(page_count):
        raw_text = raw_pages_text[i]
        page_obj = doc[i]

        # Check for tables
        tables_count = detect_tables_on_page(page_obj)
        has_table = tables_count > 0
        total_tables += tables_count

        if has_table:
            warning_msg = f"Page {i + 1}: {tables_count} table(s) detected (tagged for v2 structured parsing)"
            warnings.append(warning_msg)
            logger.debug(f"[{path.name}] {warning_msg}")

        # Clean page text
        cleaned_text = clean_page_text(raw_text, is_first_page=(i == 0))

        # Check for empty page warning
        if not cleaned_text:
            empty_msg = f"Page {i + 1}: Empty page detected (no extractable text)"
            warnings.append(empty_msg)
            logger.warning(f"[{path.name}] {empty_msg}")

        pages.append(
            PageText(
                page_number=i + 1,
                text=cleaned_text,
                raw_text=raw_text,
                char_count=len(cleaned_text),
                has_table=has_table,
            )
        )

    doc.close()

    # Combine all pages with explicit page boundary preservation
    combined_text = "\n\n".join(
        f"--- Page {p.page_number} ---\n{p.text}" for p in pages if p.text
    )
    total_chars = sum(p.char_count for p in pages)

    # Short document warning
    if total_chars < 200:
        short_msg = f"Document is very short: only {total_chars} total characters extracted."
        warnings.append(short_msg)
        logger.warning(f"[{path.name}] {short_msg}")

    return ParsedDocument(
        file_path=str(path),
        file_name=path.name,
        page_count=page_count,
        pages=pages,
        raw_text=combined_text,
        total_chars=total_chars,
        warnings=warnings,
        tables_detected=total_tables,
    )


def parse_all_pdfs(directory: str | Path = PDF_DIR) -> List[ParsedDocument]:
    """
    Parse all PDF files found in the specified directory.

    Skips any scanned PDFs with logged errors and returns parsed documents.

    Args:
        directory: Directory path containing target PDFs.

    Returns:
        List of successfully parsed ParsedDocument objects.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        logger.warning(f"Target directory does not exist: {dir_path}")
        return []

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in: {dir_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {dir_path}")
    parsed_docs: List[ParsedDocument] = []

    for pdf_path in pdf_files:
        try:
            doc = parse_pdf(pdf_path)
            parsed_docs.append(doc)
            logger.info(
                f"Successfully parsed '{pdf_path.name}': {doc.page_count} page(s), "
                f"{doc.total_chars} chars, {doc.tables_detected} table(s)"
            )
        except ScannedPDFError as e:
            logger.error(f"[EXCLUDED] {e}")
        except Exception as e:
            logger.error(f"[FAILED] Error parsing '{pdf_path.name}': {e}")

    return parsed_docs


if __name__ == "__main__":
    import sys

    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_DIR
    print("=" * 80)
    print(f"GST Circular PDF Parser - Inspection Mode")
    print(f"Scanning directory: {target_dir.resolve()}")
    print("=" * 80)

    pdf_list = sorted(target_dir.glob("*.pdf"))
    if not pdf_list:
        print(f"\nNo PDFs found in '{target_dir}'.")
        print(f"Place your GST circular PDFs into '{target_dir}' and run again.")
    else:
        results = []
        excluded = []

        for pdf in pdf_list:
            try:
                parsed = parse_pdf(pdf)
                results.append(parsed)
            except ScannedPDFError as err:
                excluded.append((pdf.name, str(err)))
            except Exception as err:
                excluded.append((pdf.name, f"Error: {err}"))

        print("\n" + "-" * 80)
        print(f"{'Filename':<40} | {'Pages':<6} | {'Chars':<8} | {'Tables':<6} | Status")
        print("-" * 80)

        for doc in results:
            status = "OK" if not doc.warnings else f"{len(doc.warnings)} warning(s)"
            print(
                f"{doc.file_name[:38]:<40} | "
                f"{doc.page_count:<6} | "
                f"{doc.total_chars:<8} | "
                f"{doc.tables_detected:<6} | "
                f"{status}"
            )
            for w in doc.warnings:
                print(f"  └── [Warning] {w}")

        if excluded:
            print("\n" + "=" * 80)
            print(f"EXCLUDED / FAILED FILES ({len(excluded)}):")
            print("-" * 80)
            for name, reason in excluded:
                print(f"  • {name}: {reason}")

        print("\n" + "=" * 80)
        print(
            f"SUMMARY: {len(results)} valid native-text doc(s) parsed, "
            f"{len(excluded)} excluded/failed."
        )
        print("=" * 80)
