"""
Section-Aware Chunker module for GST Circular RAG Assistant.

Implements the Chunk dataclass, boundary-detection regexes, and the pure
boundary detector function `detect_boundary()` tailored to Indian GST/CBIC
circular document hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS
from ingestion.pdf_parser import PageText


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


def _is_conservative_section_heading(line: str) -> bool:
    """
    Conservatively evaluate whether a line represents a section/topic heading.

    Criteria:
      - Length between 8 and 120 characters.
      - Starts with an uppercase ASCII letter.
      - Does not end with terminal sentence punctuation ('.', '?', '!', ';').
      - Does not start with common prose words, boilerplate, provisos, or statutory citations.
      - Contains at least 2 alphabetic words.
    """
    if len(line) < 8 or len(line) > 120:
        return False

    # Must start with an uppercase letter
    if not line[0].isupper():
        return False

    # Must not end with terminal sentence punctuation
    if line.endswith((".", "?", "!", ";")):
        return False

    line_lower = line.lower()

    # Reject common prose starters, boilerplate, statutory phrases
    for prefix in PROSE_HEADING_EXCLUSIONS:
        if line_lower.startswith(prefix):
            return False

    # Reject if it contains full sentence clause separators like " which " or " that " followed by verbs
    words = [w for w in re.split(r"\s+", line) if any(c.isalpha() for c in w)]
    if len(words) < 2:
        return False

    # Reject lines containing email addresses or URLs
    if "@" in line or "http://" in line or "https://" in line or "www." in line:
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

    for line in lines:
        stripped = line.strip()
        if not stripped:
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

    if flush_at_end:
        flushed = _flush_current_chunk(current_state, doc_meta)
        if flushed:
            chunks.append(flushed)

    return chunks

