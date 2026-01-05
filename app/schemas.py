from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel): # Request body for a job search query
    query: str = Field(
        ...,
        min_length=1,
        description="User search query text",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=50,
        description="Number of results to return (1–50)",
    )


class JobResult(BaseModel): # A single job result returned by the search API
    score: float
    job_title: str
    company_name: str
    job_location: str
    job_level: str
    publication_date: str | None = None
    job_category: str | None = None


class QueryResponse(BaseModel): # Response body for a job search request
    query: str
    top_k: int
    summary: str
    results: list[JobResult]
