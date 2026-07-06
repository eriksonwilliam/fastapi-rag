from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dto import IngestRequest, IngestResponse, QueryRequest, QueryResponse, Source
from app.application.rag_service import RagService
from app.domain.errors import ValidationError


def create_app(service: RagService) -> FastAPI:
    app = FastAPI(
        title="RAG Service",
        description="Servico de RAG (chunk -> embed -> retrieve -> LLM). Swagger em /docs.",
        version="0.1.0",
    )

    @app.exception_handler(ValidationError)
    async def _handle_validation(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "ValidationError", "message": str(exc)},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents", response_model=IngestResponse, status_code=201)
    def ingest(request: IngestRequest) -> IngestResponse:
        doc_id = request.doc_id or str(uuid4())
        chunks = service.ingest(doc_id, request.text)
        return IngestResponse(doc_id=doc_id, chunks=chunks)

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        answer, hits = service.query(request.question, request.top_k)
        sources = [
            Source(doc_id=hit.chunk.doc_id, text=hit.chunk.text, score=hit.score) for hit in hits
        ]
        return QueryResponse(answer=answer, sources=sources)

    return app
