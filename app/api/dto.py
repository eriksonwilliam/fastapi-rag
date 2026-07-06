from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str
    doc_id: str = Field(default="", description="Opcional; gerado se ausente")


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
