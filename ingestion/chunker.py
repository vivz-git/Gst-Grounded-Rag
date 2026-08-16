"""
Section-Aware Chunker module for GST Circular RAG Assistant.

Implements the Chunk dataclass, boundary-detection regexes, and the pure
boundary detector function `detect_boundary()` tailored to Indian GST/CBIC
circular document hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

from config.settings import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS


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
