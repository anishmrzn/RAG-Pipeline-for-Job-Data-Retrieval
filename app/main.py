from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.rag.generator import generate_grounded_answer
from app.rag.retriever import JobRetriever
from app.schemas import JobResult, QueryRequest, QueryResponse
from app.utils import sanitize_job_metadata

app = FastAPI(
    title="RAG Job Search API",
    version="0.1.0",
    description="Retrieval-based job search over the LF Jobs dataset using embeddings and FAISS.",
)

# Load heavy resources once at startup
_settings = get_settings()
_retriever = JobRetriever(_settings)


@app.get("/")
def root() -> dict:
    return {"message": "RAG Job Search API. Visit /docs for Swagger UI."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query_jobs(payload: QueryRequest) -> QueryResponse: # Retrieve top-K job listings for the given query
    try:
        results = _retriever.retrieve(payload.query, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    top_k = payload.top_k if payload.top_k is not None else _settings.top_k

    # Internal payload used only for grounded summary generation
    job_payloads: list[dict] = []
    for r in results:
        meta = dict(r.metadata)
        meta["matched_chunks"] = r.matched_chunks
        job_payloads.append(meta)

    summary = generate_grounded_answer(_settings, payload.query, job_payloads)

    clean_results = [
        JobResult(
            score=round(r.score, 3),
            **sanitize_job_metadata(r.metadata),
        )
        for r in results
    ]

    return QueryResponse(
        query=payload.query,
        top_k=top_k,
        summary=summary,
        results=clean_results,
    )
