"""
Unit tests for hierarchical section path construction and parent derivation in ingestion/chunker.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ingestion.chunker import (
    build_section_path,
    get_parent_section,
)


class TestSectionPathHierarchy(unittest.TestCase):
    """Test hierarchical section path building and state transitions."""

    def test_first_top_level_section(self):
        # "" + ("top_level", "3") -> "3"
        self.assertEqual(build_section_path("", "top_level", "3"), "3")
        self.assertEqual(build_section_path("", "top_level", "1"), "1")

    def test_top_level_reset(self):
        # "4.1.(a).(i)" + ("top_level", "5") -> "5"
        self.assertEqual(build_section_path("4.1.(a).(i)", "top_level", "5"), "5")
        self.assertEqual(build_section_path("3.1", "top_level", "4"), "4")

    def test_decimal_child(self):
        # "3" + ("decimal", "3.1") -> "3.1"
        self.assertEqual(build_section_path("3", "decimal", "3.1"), "3.1")
        self.assertEqual(build_section_path("4.1", "decimal", "4.1.2"), "4.1.2")

    def test_decimal_sibling_and_descendant_reset(self):
        # "4.1.(a)" + ("decimal", "4.2") -> "4.2"
        self.assertEqual(build_section_path("4.1.(a)", "decimal", "4.2"), "4.2")
        self.assertEqual(build_section_path("4.1.(a).(i)", "decimal", "4.2"), "4.2")

    def test_clause_child(self):
        # "4.1" + ("clause", "(a)") -> "4.1.(a)"
        self.assertEqual(build_section_path("4.1", "clause", "(a)"), "4.1.(a)")
        self.assertEqual(build_section_path("3", "clause", "(a)"), "3.(a)")

    def test_clause_sibling_replacement(self):
        # "4.1.(a)" + ("clause", "(b)") -> "4.1.(b)"
        self.assertEqual(build_section_path("4.1.(a)", "clause", "(b)"), "4.1.(b)")

    def test_clause_descendant_reset(self):
        # "4.1.(a).(i)" + ("clause", "(b)") -> "4.1.(b)"
        self.assertEqual(build_section_path("4.1.(a).(i)", "clause", "(b)"), "4.1.(b)")

    def test_roman_child(self):
        # "4.1.(a)" + ("roman", "(i)") -> "4.1.(a).(i)"
        self.assertEqual(build_section_path("4.1.(a)", "roman", "(i)"), "4.1.(a).(i)")
        self.assertEqual(build_section_path("3", "roman", "(i)"), "3.(i)")

    def test_roman_sibling_replacement(self):
        # "4.1.(a).(ii)" + ("roman", "(iii)") -> "4.1.(a).(iii)"
        self.assertEqual(build_section_path("4.1.(a).(ii)", "roman", "(iii)"), "4.1.(a).(iii)")

    def test_bullet_and_heading_preservation(self):
        # Bullet/heading preserves current structural path
        self.assertEqual(build_section_path("4.1.(a)", "bullet", "-"), "4.1.(a)")
        self.assertEqual(build_section_path("4.1.(a)", "bullet", "•"), "4.1.(a)")
        self.assertEqual(build_section_path("4.1", "heading", "Some Title"), "4.1")
        self.assertEqual(build_section_path("", "bullet", "-"), "")
        self.assertEqual(build_section_path("", "heading", "Some Title"), "")

    def test_empty_paths(self):
        self.assertEqual(build_section_path("", "top_level", "3"), "3")
        self.assertEqual(build_section_path("", "decimal", "2.1"), "2.1")
        self.assertEqual(build_section_path("", "clause", "(a)"), "(a)")
        self.assertEqual(build_section_path("", "roman", "(i)"), "(i)")

    def test_unknown_boundary_type_raises(self):
        with self.assertRaises(ValueError):
            build_section_path("4.1", "unknown_boundary", "xyz")


class TestParentSectionDerivation(unittest.TestCase):
    """Test get_parent_section derivation from section_path."""

    def test_parent_section_examples(self):
        self.assertEqual(get_parent_section("4.1.(a).(i)"), "4.1.(a)")
        self.assertEqual(get_parent_section("4.1.(a)"), "4.1")
        self.assertEqual(get_parent_section("4.1.2"), "4.1")
        self.assertEqual(get_parent_section("4.1"), "4")
        self.assertEqual(get_parent_section("4"), "")
        self.assertEqual(get_parent_section("(a)"), "")
        self.assertEqual(get_parent_section("(i)"), "")
        self.assertEqual(get_parent_section(""), "")


if __name__ == "__main__":
    unittest.main()
