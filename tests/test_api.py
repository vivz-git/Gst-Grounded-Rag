"""
Unit tests for FastAPI Application in api/app.py.
Uses FastAPI TestClient and mock RAGService to avoid real external API calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from fastapi.testclient import TestClient

from api.app import create_app
from api.service import RAGService
from generation.gemini_generator import GroundedAnswer, SourceCitation


class TestFastAPIApp(unittest.TestCase):
    """Test suite for FastAPI endpoints /health and /ask."""

    def setUp(self):
        """Set up mock service and TestClient."""
        self.mock_service = MagicMock(spec=RAGService)
        self.app = create_app(service=self.mock_service)
        self.client = TestClient(self.app)

        self.sample_sources = [
            SourceCitation(
                chunk_id="Circular-170-02-2022-GST__4.1__0010",
                source_filename="Circular-170-02-2022-GST.pdf",
                section_path="4.1",
                page_start=2,
                page_end=2,
            ),
            SourceCitation(
                chunk_id="Circular-170-02-2022-GST__6__0019",
                source_filename="Circular-170-02-2022-GST.pdf",
                section_path="6",
                page_start=7,
                page_end=7,
            ),
        ]

    def test_1_health_endpoint(self):
        """GET /health returns 200 OK with {'status': 'ok'}."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_2_valid_ask_request_grounded(self):
        """POST /ask with valid question returns 200 and grounded answer."""
        self.mock_service.ask.return_value = GroundedAnswer(
            answer="ITC reversal under Rule 42 is reported in Table 4(B)(1) of GSTR-3B.",
            confidence_label="grounded",
            sources=self.sample_sources,
            used_chunk_ids=["Circular-170-02-2022-GST__4.1__0010", "Circular-170-02-2022-GST__6__0019"],
        )

        response = self.client.post("/ask", json={"question": "How is input tax credit reversal reported?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["confidence"], "grounded")
        self.assertIn("Table 4(B)(1)", data["answer"])
        self.assertEqual(len(data["sources"]), 2)
        self.assertEqual(data["sources"][0]["chunk_id"], "Circular-170-02-2022-GST__4.1__0010")
        self.assertEqual(data["sources"][0]["source_filename"], "Circular-170-02-2022-GST.pdf")
        self.assertEqual(data["sources"][0]["section_path"], "4.1")
        self.assertEqual(data["sources"][0]["page_start"], 2)
        self.assertEqual(data["sources"][0]["page_end"], 2)

    def test_3_insufficient_evidence_response(self):
        """POST /ask returns 200 with confidence 'insufficient_evidence' when unanswerable."""
        self.mock_service.ask.return_value = GroundedAnswer(
            answer="The answer cannot be determined from the provided documents.",
            confidence_label="insufficient_evidence",
            sources=[],
            used_chunk_ids=[],
        )

        response = self.client.post("/ask", json={"question": "What is the tax rate on bitcoin?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["confidence"], "insufficient_evidence")
        self.assertEqual(data["sources"], [])
        self.assertIn("cannot be determined", data["answer"])

    def test_4_empty_question_validation(self):
        """POST /ask with empty string returns HTTP 422 Unprocessable Entity."""
        response = self.client.post("/ask", json={"question": ""})
        self.assertEqual(response.status_code, 422)

    def test_5_whitespace_question_validation(self):
        """POST /ask with only whitespace returns HTTP 422."""
        response = self.client.post("/ask", json={"question": "   \n\t  "})
        self.assertEqual(response.status_code, 422)

    def test_6_missing_question_field(self):
        """POST /ask with missing 'question' key returns HTTP 422."""
        response = self.client.post("/ask", json={"wrong_key": "some text"})
        self.assertEqual(response.status_code, 422)

    def test_7_oversized_question_validation(self):
        """POST /ask with question exceeding 1000 characters returns HTTP 422."""
        long_q = "What is GST? " * 100  # 1300 chars
        response = self.client.post("/ask", json={"question": long_q})
        self.assertEqual(response.status_code, 422)

    def test_8_retrieval_exception_handling(self):
        """Internal retrieval error returns clean HTTP 500 without leaking traceback."""
        self.mock_service.ask.side_effect = RuntimeError("Vector DB connection failed")
        response = self.client.post("/ask", json={"question": "What is Rule 36(4)?"})
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertNotIn("Traceback", data["detail"])
        self.assertNotIn("Vector DB connection failed", data["detail"])

    def test_9_generation_exception_handling(self):
        """Internal LLM generation error returns clean HTTP 500 without leaking secrets."""
        self.mock_service.ask.side_effect = Exception("Google API Key AIzaSy... quota exceeded")
        response = self.client.post("/ask", json={"question": "What is GSTR-3B?"})
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertNotIn("AIzaSy", data["detail"])
        self.assertNotIn("API Key", data["detail"])

    def test_10_extra_fields_forbidden(self):
        """Sending unexpected extra fields in request payload returns HTTP 422."""
        response = self.client.post("/ask", json={"question": "Valid query", "hack_param": "bad"})
        self.assertEqual(response.status_code, 422)

    def test_11_source_metadata_preservation(self):
        """All metadata fields in SourceCitation are faithfully preserved in JSON response."""
        self.mock_service.ask.return_value = GroundedAnswer(
            answer="Grounded answer.",
            confidence_label="grounded",
            sources=[
                SourceCitation(
                    chunk_id="c188__4.3__0006",
                    source_filename="circular-188.pdf",
                    section_path="4.3",
                    page_start=3,
                    page_end=5,
                )
            ],
        )

        response = self.client.post("/ask", json={"question": "How are refunds filed?"})
        self.assertEqual(response.status_code, 200)
        src = response.json()["sources"][0]
        self.assertEqual(src["chunk_id"], "c188__4.3__0006")
        self.assertEqual(src["source_filename"], "circular-188.pdf")
        self.assertEqual(src["section_path"], "4.3")
        self.assertEqual(src["page_start"], 3)
        self.assertEqual(src["page_end"], 5)


if __name__ == "__main__":
    unittest.main()
