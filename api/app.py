"""
FastAPI Application for GST Grounded RAG Assistant.

Exposes REST endpoints for health monitoring and legal question answering.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import AskRequest, AskResponse, HealthResponse, SourceCitationSchema
from api.service import RAGService

logger = logging.getLogger(__name__)


def create_app(service: Optional[RAGService] = None) -> FastAPI:
    """
    Factory to construct and configure the FastAPI application.

    Args:
        service: Optional pre-configured RAGService instance (useful for unit tests and mocks).

    Returns:
        Configured FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        """Manage startup and shutdown lifecycle events."""
        if getattr(application.state, "rag_service", None) is None:
            try:
                logger.info("Initializing RAGService from persisted disk indices...")
                application.state.rag_service = RAGService.from_persisted_indices()
                logger.info("RAGService initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load RAGService at startup: {e}")
                # Store None so routes can surface 503 if indices were missing on startup
                application.state.rag_service = None
        yield

    app = FastAPI(
        title="GST Grounded RAG API",
        description="Production-style API for verified question answering across Indian GST Circulars.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for local development and future frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permissive for local development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set provided service if supplied at creation time
    app.state.rag_service = service

    def get_service(request: Request) -> RAGService:
        """Dependency provider for RAGService instance."""
        rag_svc = getattr(request.app.state, "rag_service", None)
        if rag_svc is None:
            # Try on-demand initialization if not yet initialized
            try:
                rag_svc = RAGService.from_persisted_indices()
                request.app.state.rag_service = rag_svc
            except Exception as exc:
                logger.error(f"RAG service unavailable: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service is currently unavailable. Ensure retrieval indices are generated.",
                ) from exc
        return rag_svc

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        tags=["Monitoring"],
    )
    async def health() -> HealthResponse:
        """Return operational status of the service."""
        return HealthResponse(status="ok")

    @app.post(
        "/ask",
        response_model=AskResponse,
        summary="Ask Grounded GST Question",
        tags=["QA"],
    )
    async def ask(
        request: AskRequest,
        rag_service: RAGService = Depends(get_service),
    ) -> AskResponse:
        """
        Process a user question against Indian GST circulars and return a citation-grounded answer.
        """
        try:
            grounded_answer = rag_service.ask(request.question)
        except Exception as err:
            logger.error(f"Error processing question: {err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing your request. Please try again later.",
            ) from err

        sources_schema = [
            SourceCitationSchema(
                chunk_id=s.chunk_id,
                source_filename=s.source_filename,
                section_path=s.section_path,
                page_start=s.page_start,
                page_end=s.page_end,
            )
            for s in grounded_answer.sources
        ]

        return AskResponse(
            answer=grounded_answer.answer,
            confidence=grounded_answer.confidence_label,
            sources=sources_schema,
        )

    return app


# Default global application instance for uvicorn (e.g. uvicorn api.app:app)
app = create_app()
