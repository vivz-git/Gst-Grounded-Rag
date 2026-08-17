"""
Pydantic Request and Response Schemas for GST Grounded RAG API.
"""

from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceCitationSchema(BaseModel):
    """Schema representing an individual retrieved evidence citation."""
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Unique identifier of the chunk.")
    source_filename: str = Field(..., description="Originating PDF filename.")
    section_path: str = Field(..., description="Hierarchical section path (e.g. '4.1', '6').")
    page_start: int = Field(..., description="Starting page in source document (1-indexed).")
    page_end: int = Field(..., description="Ending page in source document (1-indexed).")


class AskRequest(BaseModel):
    """Schema for incoming user questions to /ask endpoint."""
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user question regarding GST circulars (max 1000 characters).",
        examples=["How is input tax credit reversal reported?"]
    )

    @field_validator("question")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        """Ensure question contains meaningful characters and is not pure whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return cleaned


class AskResponse(BaseModel):
    """Schema for grounded answer responses from /ask endpoint."""
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="Grounded response text strictly derived from evidence.")
    confidence: str = Field(..., description="Confidence categorization ('grounded' or 'insufficient_evidence').")
    sources: List[SourceCitationSchema] = Field(
        default_factory=list,
        description="Deduplicated list of source citations used as evidence."
    )


class HealthResponse(BaseModel):
    """Schema for API service health check."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Operational status of the API service.")
