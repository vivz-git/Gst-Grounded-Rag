"""
Section-Aware Chunker module for GST Circular RAG Assistant.

Implements the Chunk dataclass, boundary-detection regexes, and the pure
boundary detector function `detect_boundary()` tailored to Indian GST/CBIC
circular document hierarchy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS
from ingestion.pdf_parser import PageText, ParsedDocument

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Chunk Data Structure
# ==============================================================================

@dataclass
class Chunk:
    """
    Represents a self-contained, semantically coherent unit of legal text
    derived from a GST circular document.
    """
    # --- Identity & Source ---
    chunk_id: str
    # Unique identifier: format "{doc_id}_{section_path}_{seq:04d}"
    # e.g. "circular_172_04_2022_GST__5__0001"

    doc_id: str
    # Normalized document identifier derived from filename
    # e.g. "circular_172_04_2022_GST"

    source_filename: str
    # Original PDF filename on disk (e.g. "Circular-172-04-2022-GST.pdf")

    # --- Page & Citation Positioning ---
    page_start: int
    # 1-indexed page where this chunk begins

    page_end: int
    # 1-indexed page where this chunk ends (equals page_start for single-page chunks)

    section_path: str
    # Dot-joined hierarchical section address (e.g. "4.1", "3.(a).(i)", "5")

    parent_section: str
    # Immediate parent section_path or "" for top-level chunks

    section_title: str
    # Thematic heading governing this section, or "" if none

    # --- Text & Token Constraints ---
    text: str
    # Substantive chunk text with inherited context header prepended

    token_count: int
    # Approximate whitespace-delimited token count

    # --- Structural Classification ---
    chunk_type: str
    # One of: "header", "preamble", "paragraph", "subparagraph",
    # "clause", "roman_item", "bullet", "table_raw", "table_qa",
    # "closing", "heading_only"

    parsing_method: str = "structural"
    # Parsing strategy used (always "structural" in v1)

    # --- Domain Metadata ---
    circular_number: str = ""
    # Extracted circular reference (e.g. "172/04/2022-GST")

    date_issued: str = ""
    # Extracted issuance date in ISO format (e.g. "2022-07-06")

    metadata: dict = field(default_factory=dict)
    # Extensible metadata dictionary for future retrieval / citation signals


# ==============================================================================
# 2. Boundary Detection Patterns & Constants
# ==============================================================================

# Standard valid roman numerals used in GST circular enumerations
VALID_ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"
}

# 1. Top-Level Numbered Paragraphs: "1.", "2.", "12.", "3. Filing of refund application"
# Anchored to line start; avoids matching dates like "12.03.2021" via lookahead (?![\d])
TOP_LEVEL_REGEX = re.compile(
    r"^\s*(\d{1,2})\.(?:\s+(?![\d])(.*)|\s*)$"
)

# 2. Decimal Sub-paragraphs: "1.1", "4.2", "4.1.2", "2.1 In order to...", "4.1.2 In cases..."
# Supports nested decimal notation (e.g. "4.1", "4.1.2"); avoids 4-digit date segments
DECIMAL_SUBPARA_REGEX = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})+)\.?(?:\s+(?![\d])(.*)|\s*)$"
)

# 3. Parenthesized Roman Numerals: "(i)", "(ii)", "(iv)", "(v)", etc.
ROMAN_PAREN_REGEX = re.compile(
    r"^\s*\(((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii))\)(?:\s+(.*)|\s*)$",
    re.IGNORECASE,
)

# 4. Bare Roman Numerals with dot: "i.", "ii.", "iv.", "v."
BARE_ROMAN_REGEX = re.compile(
    r"^\s*((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii))\.(?:\s+(.*)|\s*)$",
    re.IGNORECASE,
)

# 5. Lettered Clauses: "(a)", "(b)", "(c)", "(z)"
# Single lowercase ASCII letter in parentheses
LETTERED_CLAUSE_REGEX = re.compile(
    r"^\s*\(([a-z])\)(?:\s+(.*)|\s*)$"
)

# 6. Bullets or Dash-prefixed Items: "- text", "– text", "— text", "• text", "* text"
BULLET_REGEX = re.compile(
    r"^\s*([-–—•*])\s+(.*)$"
)

# Excluded starters for section headings (prose, boilerplate, statutory references, provisos)
PROSE_HEADING_EXCLUSIONS = (
    "provided that",
    "provided further that",
    "in terms of",
    "in order to",
    "in view of",
    "in accordance with",
    "in cases where",
    "in this regard",
    "in respect of",
    "as per",
    "as specified",
    "as clarified",
    "therefore",
    "accordingly",
    "however",
    "further",
    "furthermore",
    "it is",
    "it has",
    "it may",
    "instances have",
    "where ",
    "whether ",
    "which ",
    "when ",
    "what ",
    "the ",
    "this ",
    "these ",
    "such ",
    "various ",
    "subject:",
    "subject :",
    "government of india",
    "ministry of finance",
    "department of revenue",
    "central board",
    "gst policy wing",
    "circular no",
    "f. no",
    "f.no",
    "new delhi",
    "to,",
    "madam/sir",
    "sir/madam",
    "difficulty, if any",
    "hindi version",
    "schedule i",
    "schedule ii",
    "schedule iii",
    "section ",
    "rule ",
    "table ",
    "form ",
)


# ==============================================================================
# 3. Pure Boundary Detector Function
# ==============================================================================

def detect_boundary(
    line: str,
    in_enumeration: bool = False,
    allow_bare_roman: bool = False,
) -> Optional[Tuple[str, str]]:
    """
    Detect structural section boundaries in a line of GST circular text.

    Evaluates patterns strictly in legal priority order:
      1. Top-level numbered paragraph: ("top_level", "1")
      2. Decimal sub-paragraph: ("decimal", "4.1")
      3. Parenthesized Roman numeral: ("roman", "(i)")
      4. Lettered clause: ("clause", "(a)")
      5. Bare Roman numeral (if contextual / standalone): ("roman", "i.")
      6. Dash/bullet item: ("bullet", "-")
      7. Section heading: ("heading", "Heading Text")
      8. Non-boundary: None

    Args:
        line: The text line to evaluate.
        in_enumeration: Set to True if currently parsing within an active list.
        allow_bare_roman: If True, allows bare roman numbers ("i.") with body text.

    Returns:
        A tuple of (boundary_type, marker_value) or None if the line is not a boundary.
    """
    if not line:
        return None

    stripped = line.strip()
    if not stripped:
        return None

    # --------------------------------------------------------------------------
    # Priority 1: Top-Level Numbered Paragraph ("1.", "2.", "12.", "3. Title")
    # Must precede decimal to ensure "1." is checked, but decimal regex ensures "1.1" won't match.
    # --------------------------------------------------------------------------
    # Check decimal first to prevent "4.1" or "4.1.2" matching top-level if pattern overlaps
    match_decimal = DECIMAL_SUBPARA_REGEX.match(line)
    if match_decimal:
        marker = match_decimal.group(1)
        return ("decimal", marker)

    match_top = TOP_LEVEL_REGEX.match(line)
    if match_top:
        marker = match_top.group(1)
        return ("top_level", marker)

    # --------------------------------------------------------------------------
    # Priority 2: Roman Numerals in Parentheses ("(i)", "(ii)", "(iv)", "(v)")
    # Evaluated before lettered clauses so "(i)" is categorized as roman, not clause.
    # --------------------------------------------------------------------------
    match_roman_paren = ROMAN_PAREN_REGEX.match(line)
    if match_roman_paren:
        roman = match_roman_paren.group(1).lower()
        if roman in VALID_ROMAN_NUMERALS:
            return ("roman", f"({roman})")

    # --------------------------------------------------------------------------
    # Priority 3: Lettered Clauses ("(a)", "(b)", "(c)", "(z)")
    # Single letter in parentheses.
    # --------------------------------------------------------------------------
    match_clause = LETTERED_CLAUSE_REGEX.match(line)
    if match_clause:
        clause_letter = match_clause.group(1)
        return ("clause", f"({clause_letter})")

    # --------------------------------------------------------------------------
    # Priority 4: Bare Roman Numerals with Dot ("i.", "ii.", "iv.")
    # Permitted when in enumeration, explicitly allowed, or standing alone on a line.
    # --------------------------------------------------------------------------
    match_bare_roman = BARE_ROMAN_REGEX.match(line)
    if match_bare_roman:
        roman = match_bare_roman.group(1).lower()
        if roman in VALID_ROMAN_NUMERALS:
            is_standalone = stripped == f"{roman}." or stripped == f"{match_bare_roman.group(1)}."
            if allow_bare_roman or in_enumeration or is_standalone:
                return ("roman", f"{roman}.")

    # --------------------------------------------------------------------------
    # Priority 5: Bullets / Dashes ("- text", "– text", "• text", "* text")
    # --------------------------------------------------------------------------
    match_bullet = BULLET_REGEX.match(line)
    if match_bullet:
        bullet_char = match_bullet.group(1)
        return ("bullet", bullet_char)

    # --------------------------------------------------------------------------
    # Priority 6: Conservative Section Heading
    # Standalone title-cased or sentence-cased line without terminal sentence punctuation.
    # --------------------------------------------------------------------------
    if _is_conservative_section_heading(stripped):
        return ("heading", stripped.rstrip(":").strip())

    # Not a structural boundary
    return None


# Words that, when appearing at the END of a line, indicate the line is a wrapped
# prose sentence rather than a self-contained heading.
_CONTINUATION_TAIL_WORDS: set = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "at",
    "by", "as", "is", "are", "was", "were", "be", "been", "that", "which",
    "with", "from", "into", "under", "over", "sub-", "only", "not", "no",
    "its", "their", "his", "her", "this", "these", "those", "any", "all",
    "each", "both", "than", "when", "where", "whether", "how", "may",
    "shall", "can", "will", "has", "have", "had", "also", "even", "only",
    "such", "same", "other", "another", "so", "if", "else", "but",
}


def _is_conservative_section_heading(line: str) -> bool:
    """
    Conservatively evaluate whether a line represents a genuine section/topic heading
    in a GST/CBIC circular, as opposed to a wrapped continuation sentence.

    Corpus-informed rules (from real document analysis):
      1. Length between 8 and 120 characters.
      2. Must start with an uppercase ASCII letter.
      3. Must NOT end with terminal sentence punctuation ('.' '?' '!' ';').
      4. Must NOT end with a hyphen (line-wrapped mid-word or mid-year like '2018-').
      5. Must NOT end with a trailing comma (prose continuation marker).
      6. Must NOT end with a continuation/function word (the, and, of, in, a, etc.)
         — these indicate the sentence wrapped to the next line.
      7. Must NOT start with any prose/boilerplate/statutory prefix.
      8. Must NOT start with a lowercase letter.
      9. Must NOT contain mid-line sentence punctuation followed by a new clause
         (period + space + uppercase = two-sentence prose, not a heading).
      10. Must contain at least 3 alphabetic words.
      11. Must NOT match known boilerplate identifier patterns
          (e.g. 'Principal Commissioner', 'Sanjay Mangal', bare months, bare years).
      12. Must NOT contain a slash that suggests an address or option ('(All)', 'he/she').
    """
    if len(line) < 8 or len(line) > 120:
        return False

    stripped = line.strip()
    if not stripped:
        return False

    # Rule 2: Must start with an uppercase ASCII letter
    if not stripped[0].isupper():
        return False

    # Rule 3: Must not end with terminal sentence punctuation
    if stripped.endswith((".", "?", "!", ";")):
        return False

    # Rule 4: Must not end with a hyphen (wrapped mid-word like 'sub-' or mid-year '2018-')
    if stripped.endswith("-"):
        return False

    # Rule 5: Must not end with a trailing comma (prose continuation)
    if stripped.endswith(","):
        return False

    # Rule 6: Must not end with a continuation/function word
    last_word = stripped.rstrip(".,;:)-").rsplit(None, 1)[-1].lower().rstrip(")-,.") if stripped.rsplit(None, 1) else ""
    if last_word in _CONTINUATION_TAIL_WORDS:
        return False

    line_lower = stripped.lower()

    # Rule 7: Reject common prose starters, boilerplate, statutory phrases
    for prefix in PROSE_HEADING_EXCLUSIONS:
        if line_lower.startswith(prefix):
            return False

    # Rule 9: Must not contain mid-line sentence punctuation ('. ' followed by uppercase = 2+ sentences)
    if re.search(r"\. [A-Z]", stripped):
        return False

    # Rule 10: Require at least 3 alphabetic words
    words = [w for w in re.split(r"\s+", stripped) if any(c.isalpha() for c in w)]
    if len(words) < 3:
        return False

    # Reject lines containing email addresses or URLs
    if "@" in stripped or "http://" in stripped or "https://" in stripped or "www." in stripped:
        return False

    # Rule 11: Reject single-name or title-only lines, including with parenthetical qualifiers.
    # Patterns: 'Sanjay Mangal', 'Principal Commissioner (GST)', 'Director General (DRI)'
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+(\s*\([^)]*\))?\s*$", stripped):
        return False
    # Month name at start indicates a date/sentence fragment
    month_pat = r"^(January|February|March|April|May|June|July|August|September|October|November|December)"
    if re.match(month_pat, stripped):
        return False

    # Rule 12: Lines containing '(All)' or slash-based options are addressee fragments
    if "(All)" in stripped or "/" in stripped:
        return False

    return True


# ==============================================================================
# 4. Hierarchical Section Path Construction & Parent Derivation
# ==============================================================================

def _strip_trailing_token(path: str) -> Tuple[str, str]:
    """
    Split the rightmost dot-separated segment from path.
    Returns (remaining_path, last_token).
    """
    if not path:
        return "", ""
    if "." in path:
        prefix, _, token = path.rpartition(".")
        return prefix, token
    return "", path


def _is_roman_token(token: str) -> bool:
    """Check if token represents a parenthesized or bare roman numeral."""
    if not token:
        return False
    t = token.lower().strip()
    return (
        t in {f"({r})" for r in VALID_ROMAN_NUMERALS}
        or t in {f"{r}." for r in VALID_ROMAN_NUMERALS}
        or t in VALID_ROMAN_NUMERALS
    )


def _is_clause_token(token: str) -> bool:
    """Check if token represents a single-letter clause (e.g. '(a)', '(b)')."""
    if not token:
        return False
    t = token.lower().strip()
    return bool(re.match(r"^\([a-z]\)$", t)) and not _is_roman_token(token)


def build_section_path(
    current_path: str,
    boundary_type: str,
    marker_value: str,
) -> str:
    """
    Construct a deterministic hierarchical section path given the current path
    and a detected section boundary.

    Hierarchy Rules:
      - 'top_level': Replaces the entire hierarchy with the top-level marker (e.g. '3').
      - 'decimal': Replaces with the decimal marker (e.g. '4.1', '4.2', '4.1.2').
      - 'clause': Strips any descendant roman/bullet and previous sibling clause,
                  then appends the new clause marker (e.g. '4.1.(a)', '4.1.(b)').
      - 'roman': Strips previous sibling roman numeral and appends the new roman marker.
      - 'bullet': Preserves current section path without altering hierarchy.
      - 'heading': Preserves current section path without altering hierarchy.

    Args:
        current_path: Active section path before this boundary (e.g. "4.1.(a).(i)").
        boundary_type: One of 'top_level', 'decimal', 'clause', 'roman', 'bullet', 'heading'.
        marker_value: The extracted marker string (e.g. '5', '4.2', '(a)', '(i)', '-').

    Returns:
        New deterministic section_path string.

    Raises:
        ValueError: If boundary_type is unrecognized.
    """
    b_type = boundary_type.lower().strip()

    if b_type == "top_level":
        return marker_value.strip()

    if b_type == "decimal":
        return marker_value.strip()

    if b_type == "clause":
        base = current_path.strip()
        # 1. Strip trailing roman token if present (descendant reset)
        prefix, token = _strip_trailing_token(base)
        if _is_roman_token(token):
            base = prefix
        # 2. Strip trailing clause token if present (sibling replacement)
        prefix, token = _strip_trailing_token(base)
        if _is_clause_token(token):
            base = prefix
        # 3. Append clause
        clause_str = marker_value.strip()
        if not clause_str.startswith("("):
            clause_str = f"({clause_str})"
        return f"{base}.{clause_str}" if base else clause_str

    if b_type == "roman":
        base = current_path.strip()
        # 1. Strip trailing roman token if present (sibling replacement)
        prefix, token = _strip_trailing_token(base)
        if _is_roman_token(token):
            base = prefix
        # 2. Append roman
        roman_str = marker_value.strip()
        return f"{base}.{roman_str}" if base else roman_str

    if b_type in ("bullet", "heading"):
        return current_path.strip()

    raise ValueError(f"Unknown boundary type: {boundary_type!r}")


def get_parent_section(section_path: str) -> str:
    """
    Derive the immediate parent section path from a hierarchical section_path.

    Examples:
        "4.1.(a).(i)" -> "4.1.(a)"
        "4.1.(a)"     -> "4.1"
        "4.1.2"       -> "4.1"
        "4.1"         -> "4"
        "4"           -> ""
        "(a)"         -> ""
        ""            -> ""
    """
    if not section_path:
        return ""
    if "." in section_path:
        return section_path.rpartition(".")[0]
    return ""


# ==============================================================================
# 5. Chunker State & Page-Level Processing
# ==============================================================================

@dataclass
class ChunkerState:
    """
    Maintains active hierarchical context, metadata, and line buffers across lines
    and page boundaries during document chunking.
    """
    current_section_path: str = ""
    current_section_title: str = ""
    current_chunk_type: str = "paragraph"
    current_lines: List[str] = field(default_factory=list)
    chunk_page_start: int = 1
    chunk_page_end: int = 1
    sequence_number: int = 1
    in_enumeration: bool = False
    trailing_hyphen_active: bool = False
    # Set to True when the last substantive line of the previous page ended with a
    # trailing hyphen, indicating a mid-word or mid-token split across a page break
    # (e.g. '2018-' on page N → '19. ...' on page N+1 should NOT be treated as Para 19).


def _flush_current_chunk(
    state: ChunkerState,
    doc_meta: Dict[str, Any],
) -> Optional[Chunk]:
    """
    Construct a Chunk from the current buffer in ChunkerState and reset the buffer.
    Returns None if the buffer is empty or contains only an isolated marker line.
    """
    if not state.current_lines:
        return None

    raw_body = "\n".join(state.current_lines).strip()
    if not raw_body:
        state.current_lines = []
        return None

    # Check if raw_body is only an isolated marker line (e.g. "1." or "4.1" or "(a)")
    words = [w for w in re.split(r"\s+", raw_body) if any(c.isalnum() for c in w)]
    if len(words) == 0:
        state.current_lines = []
        return None

    if len(words) == 1 and (
        words[0].rstrip(".").isdigit()
        or _is_roman_token(words[0])
        or _is_clause_token(words[0])
    ):
        state.current_lines = []
        return None

    doc_id = doc_meta.get("doc_id", "doc")
    source_filename = doc_meta.get("source_filename", f"{doc_id}.pdf")
    circular_number = doc_meta.get("circular_number", "")
    date_issued = doc_meta.get("date_issued", "")

    section_path = state.current_section_path or "1"
    parent_section = get_parent_section(section_path)
    section_title = state.current_section_title or ""

    # Context header format: [{circular_number} | {section_path} | {section_title}]
    context_header = f"[{circular_number} | {section_path} | {section_title}]"
    full_text = f"{context_header}\n{raw_body}"
    token_count = len(full_text.split())

    # Deterministic chunk_id: format "{doc_id}_{clean_section_path}_{seq:04d}"
    clean_path = re.sub(r"[^a-zA-Z0-9_.]", "_", section_path)
    chunk_id = f"{doc_id}__{clean_path}__{state.sequence_number:04d}"
    state.sequence_number += 1

    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_start=state.chunk_page_start,
        page_end=state.chunk_page_end,
        section_path=section_path,
        parent_section=parent_section,
        section_title=section_title,
        text=full_text,
        token_count=token_count,
        chunk_type=state.current_chunk_type,
        parsing_method="structural",
        circular_number=circular_number,
        date_issued=date_issued,
    )

    state.current_lines = []
    return chunk


def flush_chunker_state(
    state: ChunkerState,
    doc_meta: Dict[str, Any],
) -> Optional[Chunk]:
    """
    Public function to flush any remaining pending chunk buffer in ChunkerState.
    """
    return _flush_current_chunk(state, doc_meta)


def chunk_page(
    page: PageText,
    doc_meta: Dict[str, Any],
    current_state: ChunkerState,
    flush_at_end: bool = False,
) -> List[Chunk]:
    """
    Process a single PageText line-by-line and assemble structural chunks.

    Args:
        page: The PageText object containing page number, cleaned text, and table flag.
        doc_meta: Document metadata dictionary (doc_id, source_filename, circular_number, date_issued).
        current_state: Mutable ChunkerState tracking hierarchy and buffer across pages.
        flush_at_end: If True, flushes any remaining line buffer at the end of this page.

    Returns:
        List of completed Chunk objects produced on this page.
    """
    chunks: List[Chunk] = []

    # --------------------------------------------------------------------------
    # 1. Table Handling: Atomic Single-Chunk Emission
    # --------------------------------------------------------------------------
    if page.has_table:
        # Flush any existing buffer before table
        pre_chunk = _flush_current_chunk(current_state, doc_meta)
        if pre_chunk:
            chunks.append(pre_chunk)

        doc_id = doc_meta.get("doc_id", "doc")
        source_filename = doc_meta.get("source_filename", f"{doc_id}.pdf")
        circular_number = doc_meta.get("circular_number", "")
        date_issued = doc_meta.get("date_issued", "")

        # Check for strong Q&A indicators
        text_lower = page.text.lower()
        is_qa = "whether" in text_lower or text_lower.startswith("q") or " if " in text_lower
        chunk_type = "table_qa" if is_qa else "table_raw"

        section_path = current_state.current_section_path or f"p.{page.page_number}"
        parent_section = get_parent_section(section_path)
        section_title = current_state.current_section_title or ""

        context_header = f"[{circular_number} | {section_path} | {section_title}]"
        full_text = f"{context_header}\n{page.text}"
        token_count = len(full_text.split())

        clean_path = re.sub(r"[^a-zA-Z0-9_.]", "_", section_path)
        chunk_id = f"{doc_id}__{clean_path}__{current_state.sequence_number:04d}"
        current_state.sequence_number += 1

        table_chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_filename=source_filename,
            page_start=page.page_number,
            page_end=page.page_number,
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
        chunks.append(table_chunk)
        return chunks

    # --------------------------------------------------------------------------
    # 2. Regular Text Processing Line-by-Line
    # --------------------------------------------------------------------------
    lines = page.text.splitlines()

    # Bug Fix 2: Identify if the first substantive line on this page should be
    # treated as a continuation because the previous page ended with a trailing hyphen.
    # If so, join the first substantive line as continuation text, skip boundary detection
    # on it, and clear the flag.
    first_substantive_consumed = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Bug Fix 2: If the previous page ended with a trailing hyphen (e.g. "2018-")
        # then this first substantive line is a cross-page continuation, NOT a new
        # structural boundary — even if it looks like "19. Further...".
        if current_state.trailing_hyphen_active and not first_substantive_consumed:
            first_substantive_consumed = True
            current_state.trailing_hyphen_active = False
            # Treat as pure continuation regardless of boundary detector result
            if not current_state.current_section_path:
                current_state.current_section_path = "preamble" if page.page_number == 1 else "1"
                current_state.current_chunk_type = "preamble" if page.page_number == 1 else "paragraph"
                current_state.chunk_page_start = page.page_number
            current_state.chunk_page_end = page.page_number
            current_state.current_lines.append(line)
            continue

        first_substantive_consumed = True

        # Bug Fix 2 (intra-page): If the last line already added to the buffer
        # ends with a trailing hyphen, the current line continues that token/word
        # and must NOT be treated as a new boundary — even if it starts with "19."
        prev_line_in_buffer = current_state.current_lines[-1].strip() if current_state.current_lines else ""
        if prev_line_in_buffer.endswith("-"):
            if not current_state.current_section_path:
                current_state.current_section_path = "preamble" if page.page_number == 1 else "1"
                current_state.current_chunk_type = "preamble" if page.page_number == 1 else "paragraph"
                current_state.chunk_page_start = page.page_number
            current_state.chunk_page_end = page.page_number
            current_state.current_lines.append(line)
            continue

        boundary = detect_boundary(line, in_enumeration=current_state.in_enumeration)

        if boundary is not None:
            b_type, b_marker = boundary

            if b_type == "heading":
                # Heading line: flush active chunk, update title
                flushed = _flush_current_chunk(current_state, doc_meta)
                if flushed:
                    chunks.append(flushed)
                current_state.current_section_title = b_marker
                continue

            # Structural marker (top_level, decimal, clause, roman, bullet)
            flushed = _flush_current_chunk(current_state, doc_meta)
            if flushed:
                chunks.append(flushed)

            current_state.in_enumeration = (b_type in ("clause", "roman", "bullet"))
            new_path = build_section_path(current_state.current_section_path, b_type, b_marker)
            current_state.current_section_path = new_path
            current_state.chunk_page_start = page.page_number
            current_state.chunk_page_end = page.page_number

            type_map = {
                "top_level": "paragraph",
                "decimal": "subparagraph",
                "clause": "clause",
                "roman": "roman_item",
                "bullet": "bullet",
            }
            current_state.current_chunk_type = type_map.get(b_type, "paragraph")
            current_state.current_lines.append(line)

        else:
            # Continuation line (proviso, ordinary prose, table fragment, etc.)
            if not current_state.current_section_path:
                current_state.current_section_path = "preamble" if page.page_number == 1 else "1"
                current_state.current_chunk_type = "preamble" if page.page_number == 1 else "paragraph"
                current_state.chunk_page_start = page.page_number

            current_state.chunk_page_end = page.page_number
            current_state.current_lines.append(line)

    # Bug Fix 2: Before leaving this page, check if the last substantive line ends
    # with a hyphen. If so, arm the flag so the next page's first line is treated as
    # continuation regardless of its structural appearance.
    last_substantive = ""
    for l in reversed(lines):
        if l.strip():
            last_substantive = l.strip()
            break
    if last_substantive.endswith("-"):
        current_state.trailing_hyphen_active = True

    if flush_at_end:
        flushed = _flush_current_chunk(current_state, doc_meta)
        if flushed:
            chunks.append(flushed)

    return chunks


# ==============================================================================
# 6. Small-Fragment Merging
# ==============================================================================

def _extract_substantive_text(text: str) -> str:
    """
    Extract body text excluding the first line context header if present.
    """
    lines = text.split("\n", 1)
    if len(lines) > 1 and lines[0].startswith("[") and " | " in lines[0] and lines[0].endswith("]"):
        return lines[1].strip()
    return text.strip()


def _are_chunks_compatible(c1: Chunk, c2: Chunk) -> bool:
    """
    Determine if two chunks are structurally compatible to be merged.

    Rules:
      - Table chunks cannot merge with any chunk.
      - Chunks must belong to the same document.
      - Same section path (e.g. both '4.1' or both 'preamble') -> compatible.
      - Both preamble/header -> compatible.
      - Parent-child hierarchy (e.g. '4.1' and '4.1.(a)', or '3' and '3.1') -> compatible.
      - Sibling sub-sections under the same non-empty parent (e.g. '4.1.(a)' and '4.1.(b)') -> compatible.
      - Unrelated top-level paragraphs (e.g. '5' and '6') -> NOT compatible.
    """
    # Table chunks cannot be merged with other chunks
    if c1.chunk_type.startswith("table") or c2.chunk_type.startswith("table"):
        return False

    # Must be from the same document
    if c1.doc_id != c2.doc_id:
        return False

    # Same section path (e.g. both '4.1' or both 'preamble')
    if c1.section_path and c1.section_path == c2.section_path:
        return True

    # Preamble / header compatibility
    if c1.chunk_type in ("preamble", "header") and c2.chunk_type in ("preamble", "header"):
        return True

    # Parent-child relationship
    if c1.parent_section and c1.parent_section == c2.section_path:
        return True
    if c2.parent_section and c2.parent_section == c1.section_path:
        return True

    # Sibling clauses under same non-empty parent (e.g. '4.1.(a)' and '4.1.(b)')
    if c1.parent_section and c2.parent_section and c1.parent_section == c2.parent_section:
        return True

    # Prefix hierarchy (e.g. '3' and '3.1', or '4.1' and '4.1.2')
    if c1.section_path and c2.section_path:
        if c2.section_path.startswith(f"{c1.section_path}."):
            return True
        if c1.section_path.startswith(f"{c2.section_path}."):
            return True

    return False


def _merge_two_chunks(target: Chunk, frag: Chunk, frag_is_before: bool) -> Chunk:
    """
    Merge `frag` chunk into `target` chunk, preserving `target`'s metadata.
    If `frag_is_before` is True, frag's body text is prepended before target's body text.
    Otherwise, frag's body text is appended after target's body text.
    """
    target_body = _extract_substantive_text(target.text)
    frag_body = _extract_substantive_text(frag.text)

    if frag_is_before:
        merged_body = f"{frag_body}\n{target_body}" if frag_body else target_body
    else:
        merged_body = f"{target_body}\n{frag_body}" if target_body else frag_body

    context_header = f"[{target.circular_number} | {target.section_path} | {target.section_title}]"
    full_text = f"{context_header}\n{merged_body}" if merged_body else context_header
    token_count = len(full_text.split())

    page_start = min(target.page_start, frag.page_start)
    page_end = max(target.page_end, frag.page_end)

    return Chunk(
        chunk_id=target.chunk_id,
        doc_id=target.doc_id,
        source_filename=target.source_filename,
        page_start=page_start,
        page_end=page_end,
        section_path=target.section_path,
        parent_section=target.parent_section,
        section_title=target.section_title,
        text=full_text,
        token_count=token_count,
        chunk_type=target.chunk_type,
        parsing_method=target.parsing_method,
        circular_number=target.circular_number,
        date_issued=target.date_issued,
        metadata=dict(target.metadata),
    )


def _can_merge(target: Chunk, frag: Chunk, frag_is_before: bool) -> bool:
    """
    Check if `frag` can be safely merged into `target` without violating
    compatibility or MAX_CHUNK_TOKENS constraint.
    """
    if not _are_chunks_compatible(target, frag):
        return False
    merged = _merge_two_chunks(target, frag, frag_is_before)
    return merged.token_count <= MAX_CHUNK_TOKENS


def merge_small_fragments(chunks: List[Chunk]) -> List[Chunk]:
    """
    Post-processing pass to merge small chunk fragments (< MIN_CHUNK_TOKENS)
    into adjacent compatible chunks.

    Rules:
      1. Prefer merging upward into immediately preceding compatible chunk.
      2. If no preceding compatible chunk exists, merge downward into immediately
         following compatible chunk.
      3. Recompute token_count, page_start, and page_end.
      4. Never merge across incompatible boundaries (e.g. table chunks, unrelated sections).
      5. Never exceed MAX_CHUNK_TOKENS.
      6. If no safe merge is possible, retain the fragment rather than corrupting structure.
      7. Deterministically re-index chunk_ids at the end of the pass.

    Args:
        chunks: List of Chunk objects to post-process.

    Returns:
        New list of Chunk objects with small fragments merged where compatible.
    """
    if not chunks:
        return []

    # Pass 1: Upward merge pass (prefer merging into preceding compatible chunk)
    merged_list: List[Chunk] = []
    for c in chunks:
        if not merged_list:
            merged_list.append(c)
            continue

        prev = merged_list[-1]
        if c.token_count < MIN_CHUNK_TOKENS and _can_merge(prev, c, frag_is_before=False):
            merged_list[-1] = _merge_two_chunks(prev, c, frag_is_before=False)
        else:
            merged_list.append(c)

    # Pass 2: Downward merge pass for any remaining small chunks
    # (e.g. first chunk was small and had no preceding chunk, or upward merge was incompatible)
    final_list: List[Chunk] = []
    i = 0
    while i < len(merged_list):
        curr = merged_list[i]
        if curr.token_count < MIN_CHUNK_TOKENS and (i + 1) < len(merged_list):
            next_chunk = merged_list[i + 1]
            if _can_merge(next_chunk, curr, frag_is_before=True):
                # Merge curr downward into next_chunk
                merged_next = _merge_two_chunks(next_chunk, curr, frag_is_before=True)
                merged_list[i + 1] = merged_next
                i += 1
                continue
        final_list.append(curr)
        i += 1

    # Pass 3: Re-generate deterministic chunk IDs with clean 1-based sequential numbering
    result: List[Chunk] = []
    for seq, c in enumerate(final_list, start=1):
        clean_path = re.sub(r"[^a-zA-Z0-9_.]", "_", c.section_path)
        new_id = f"{c.doc_id}__{clean_path}__{seq:04d}"
        c_updated = Chunk(
            chunk_id=new_id,
            doc_id=c.doc_id,
            source_filename=c.source_filename,
            page_start=c.page_start,
            page_end=c.page_end,
            section_path=c.section_path,
            parent_section=c.parent_section,
            section_title=c.section_title,
            text=c.text,
            token_count=c.token_count,
            chunk_type=c.chunk_type,
            parsing_method=c.parsing_method,
            circular_number=c.circular_number,
            date_issued=c.date_issued,
            metadata=dict(c.metadata),
        )
        result.append(c_updated)

    return result


# ==============================================================================
# 7. Oversized Chunk Splitting
# ==============================================================================

def _split_into_structural_blocks(text: str) -> List[str]:
    """
    Split substantive body text into coherent structural blocks:
      - Lettered clauses: (a), (b), (c)
      - Roman items: (i), (ii), (iii), i., ii.
      - Dash/bullet items: -, –, —, •, *

    Returns a list of coherent structural text blocks. If no structural markers
    are found, returns [text].
    """
    lines = text.splitlines()
    if not lines:
        return []

    blocks: List[List[str]] = []
    current_block: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                current_block.append(line)
            continue

        # Check if line starts a structural marker
        is_marker = (
            bool(re.match(r"^\s*\([a-z]\)\s+", line))
            or bool(re.match(r"^\s*\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\)\s+", line, re.IGNORECASE))
            or bool(re.match(r"^\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\.\s+", line, re.IGNORECASE))
            or bool(re.match(r"^\s*[-–—•*]\s+", line))
        )

        if is_marker and current_block:
            blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return ["\n".join(b).strip() for b in blocks if "\n".join(b).strip()]


def _split_into_legal_sentences(text: str) -> List[str]:
    """
    Split text into legal sentences while strictly preserving:
      - Proviso clauses ("Provided that...", "Provided further that...")
      - Statutory citations ("Section 17(5)", "Rule 36(4)", "FORM GSTR-3B")
      - Abbreviations ("e.g.", "i.e.", "etc.", "w.e.f.", "No.", "Rs.")
      - Number/year/date formats ("2018-19", "12.03.2021", "5 lakh")
    """
    if not text.strip():
        return []

    # Protected abbreviations that end with a dot but do NOT terminate a sentence
    abbrev_pattern = (
        r"(?i)\b("
        r"sec|section|sub-sec|sub-section|rule|sub-rule|clause|sub-clause|"
        r"no|f\.no|f\. no|circ|circular|notif|notification|"
        r"rs|dr|mr|mrs|ms|ca|cma|adv|"
        r"e\.g|i\.e|viz|etc|al|w\.e\.f|r\.w|u/s|dt|dtd|dated|"
        r"form|gstr|gstin|udin"
        r")\.$"
    )

    # Protect decimal numbers before dot (e.g. "3.2", "12.03.2021", "5.0")
    decimal_pattern = r"\d+\.$"

    # Find potential sentence boundaries: dot, question mark, exclamation followed by space/newline
    raw_splits = re.split(r"(?<=[.?!])\s+(?=[A-Z\"'(])", text)
    if len(raw_splits) <= 1:
        return [text.strip()]

    sentences: List[str] = []
    current_sentence = ""

    for part in raw_splits:
        part_str = part.strip()
        if not part_str:
            continue

        if not current_sentence:
            current_sentence = part_str
            continue

        # Check if the boundary between current_sentence and part_str was a false positive
        last_word_match = re.search(r"(\S+)$", current_sentence)
        last_word = last_word_match.group(1) if last_word_match else ""

        # False boundary checks
        is_false_boundary = (
            bool(re.search(abbrev_pattern, last_word))
            or bool(re.search(decimal_pattern, last_word))
            or last_word.endswith(("vs.", "v.", "fig.", "ref."))
            or bool(re.search(r"\bFY\s+\d{4}-\d{2}\.$", current_sentence))
            or bool(re.search(r"\b\d{1,2}\.\d{1,2}\.\d{4}\.$", current_sentence))
            or bool(re.search(r"\bSection\s+\d+(\(\w+\))*\.$", current_sentence, re.IGNORECASE))
        )

        # Proviso protection: If current_sentence contains an incomplete proviso clause
        if re.search(r"(?i)\bProvided\s+(?:further\s+|also\s+)?that\b", current_sentence) and not current_sentence.endswith("."):
            is_false_boundary = True

        if is_false_boundary:
            current_sentence = f"{current_sentence} {part_str}"
        else:
            sentences.append(current_sentence)
            current_sentence = part_str

    if current_sentence:
        sentences.append(current_sentence)

    return [s.strip() for s in sentences if s.strip()]


def _split_single_oversized_chunk(chunk: Chunk) -> List[Chunk]:
    """
    Split a single oversized Chunk (> MAX_CHUNK_TOKENS) into smaller coherent child chunks.
    """
    body = _extract_substantive_text(chunk.text)
    if not body:
        return [chunk]

    # Context header overhead calculation
    dummy_header = f"[{chunk.circular_number} | {chunk.section_path}[1] | {chunk.section_title}]"
    header_tokens = len(dummy_header.split())
    max_body_tokens = max(100, MAX_CHUNK_TOKENS - header_tokens)

    # Step 1: Try structural block splitting
    structural_blocks = _split_into_structural_blocks(body)

    atomic_units: List[str] = []
    if len(structural_blocks) > 1:
        # If any block is larger than max_body_tokens, break that block into sentences
        for block in structural_blocks:
            if len(block.split()) > max_body_tokens:
                atomic_units.extend(_split_into_legal_sentences(block))
            else:
                atomic_units.append(block)
    else:
        # No structural markers, break body into legal sentences
        atomic_units = _split_into_legal_sentences(body)

    # If text could not be broken into more than 1 unit, it's unsplittable
    if len(atomic_units) <= 1:
        logger.warning(
            f"Chunk {chunk.chunk_id} ({chunk.token_count} tokens) could not be safely split "
            f"without violating legal context constraints."
        )
        return [chunk]

    # Step 2: Pack atomic units into child groups
    child_groups: List[List[str]] = []
    current_group: List[str] = []
    current_body_tokens = 0

    for unit in atomic_units:
        u_tokens = len(unit.split())
        if not current_group:
            current_group.append(unit)
            current_body_tokens = u_tokens
        elif current_body_tokens + u_tokens <= max_body_tokens:
            current_group.append(unit)
            current_body_tokens += u_tokens
        else:
            child_groups.append(current_group)
            current_group = [unit]
            current_body_tokens = u_tokens

    if current_group:
        child_groups.append(current_group)

    # Step 3: Rebalance tail fragment if last group is tiny (< MIN_CHUNK_TOKENS)
    if len(child_groups) >= 2:
        last_tokens = sum(len(u.split()) for u in child_groups[-1]) + header_tokens
        if last_tokens < MIN_CHUNK_TOKENS:
            prev_group = child_groups[-2]
            if len(prev_group) > 1:
                shifted_unit = prev_group[-1]
                shifted_tokens = len(shifted_unit.split())
                if (last_tokens + shifted_tokens) <= MAX_CHUNK_TOKENS:
                    prev_group.pop()
                    child_groups[-1].insert(0, shifted_unit)

    # Step 4: Construct child Chunk objects
    child_chunks: List[Chunk] = []
    for k, group in enumerate(child_groups, start=1):
        child_body = "\n".join(group).strip()
        child_section_path = f"{chunk.section_path}[{k}]"
        child_header = f"[{chunk.circular_number} | {child_section_path} | {chunk.section_title}]"
        child_text = f"{child_header}\n{child_body}"
        child_tokens = len(child_text.split())

        clean_path = re.sub(r"[^a-zA-Z0-9_.]", "_", child_section_path)
        child_id = f"{chunk.doc_id}__{clean_path}__{k:04d}"

        child_chunk = Chunk(
            chunk_id=child_id,
            doc_id=chunk.doc_id,
            source_filename=chunk.source_filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=child_section_path,
            parent_section=chunk.parent_section,
            section_title=chunk.section_title,
            text=child_text,
            token_count=child_tokens,
            chunk_type=chunk.chunk_type,
            parsing_method=chunk.parsing_method,
            circular_number=chunk.circular_number,
            date_issued=chunk.date_issued,
            metadata=dict(chunk.metadata),
        )
        child_chunks.append(child_chunk)

    return child_chunks


def split_oversized_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Split oversized chunks (> MAX_CHUNK_TOKENS) into smaller coherent chunks
    at structural or sentence boundaries.

    Rules:
      1. Table chunks (table_raw, table_qa) are never split (atomic v1 rule).
      2. Chunks <= MAX_CHUNK_TOKENS are returned unchanged.
      3. Oversized chunks are split at structural boundaries (clauses, roman numerals, bullets)
         first, then at sentence boundaries.
      4. Never split inside provisos, statutory citations, year/date tokens, or single sentences.
      5. Each split child inherits context header with indexed section_path (e.g. '4[1]', '4[2]').
      6. Token counts and metadata are deterministically recomputed.
      7. If a chunk cannot be safely split without violating constraints, it is kept intact.

    Args:
        chunks: List of Chunk objects to evaluate.

    Returns:
        List of Chunk objects with oversized chunks split where possible.
    """
    if not chunks:
        return []

    result: List[Chunk] = []
    for c in chunks:
        # Table chunks are never split
        if c.chunk_type.startswith("table"):
            result.append(c)
            continue

        # Chunks within token limits are never split
        if c.token_count <= MAX_CHUNK_TOKENS:
            result.append(c)
            continue

        # Oversized chunk: split
        split_pieces = _split_single_oversized_chunk(c)
        result.extend(split_pieces)

    return result


# ==============================================================================
# 8. Document-Level Chunking Orchestration
# ==============================================================================

def chunk_document(
    parsed_doc: ParsedDocument,
    doc_metadata: Dict[str, Any],
) -> List[Chunk]:
    """
    Orchestrates end-to-end document chunking for a ParsedDocument.

    Workflow:
      1. Validates input ParsedDocument and doc_metadata.
      2. Iterates through pages with a persistent ChunkerState across page boundaries.
      3. Calls chunk_page(page, doc_metadata, state, flush_at_end=False) for each page.
      4. Flushes any remaining open section state exactly once at the end of the document.
      5. Applies merge_small_fragments() to combine tiny sub-threshold fragments safely.
      6. Applies split_oversized_chunks() to ensure non-table chunks fit within MAX_CHUNK_TOKENS.
      7. Returns the final deterministic list of Chunk objects in document order.

    Args:
        parsed_doc: The ParsedDocument produced by pdf_parser.
        doc_metadata: Metadata dictionary containing 'doc_id', 'source_filename',
                      'circular_number', and optional 'date_issued'.

    Returns:
        List of finalized, deterministic Chunk objects in document order.

    Raises:
        TypeError: If parsed_doc is not a ParsedDocument or doc_metadata is not a dict.
        ValueError: If required metadata keys are missing.
    """
    if not isinstance(parsed_doc, ParsedDocument):
        raise TypeError(f"Expected ParsedDocument instance, got {type(parsed_doc).__name__}")

    if not isinstance(doc_metadata, dict):
        raise TypeError(f"Expected doc_metadata to be a dict, got {type(doc_metadata).__name__}")

    # Validate required metadata fields
    required_keys = {"doc_id", "source_filename", "circular_number"}
    missing = required_keys - set(doc_metadata.keys())
    if missing:
        raise ValueError(f"Missing required metadata fields in doc_metadata: {sorted(missing)}")

    if not parsed_doc.pages:
        return []

    state = ChunkerState()
    raw_chunks: List[Chunk] = []

    # 1. Process pages line-by-line across page boundaries
    for page in parsed_doc.pages:
        page_chunks = chunk_page(page, doc_metadata, state, flush_at_end=False)
        raw_chunks.extend(page_chunks)

    # 2. Flush remaining state at the very end of the document
    final_chunk = flush_chunker_state(state, doc_metadata)
    if final_chunk is not None:
        raw_chunks.append(final_chunk)

    if not raw_chunks:
        return []

    # 3. Merge small fragments (< MIN_CHUNK_TOKENS) into compatible neighbors
    merged_chunks = merge_small_fragments(raw_chunks)

    # 4. Split oversized non-table chunks (> MAX_CHUNK_TOKENS)
    final_chunks = split_oversized_chunks(merged_chunks)

    return final_chunks




