from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from bs4 import BeautifulSoup

from app.constants import DatasetColumns


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class JobTextInput:
    job_title: str
    company_name: str
    location: str
    job_level: str
    tags: str
    job_description_html: str


def normalize_whitespace(text: str) -> str: #repeated whitespaces into single spaces
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def html_to_text(html: str) -> str: #html to plain text
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for node in soup(["script", "style"]):
        node.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return normalize_whitespace(text)


def chunk_text(text: str, *, max_chars: int = 900, overlap: int = 150) -> list[str]: #Split text into overlapping character chunks
    clean = normalize_whitespace(text)
    if not clean:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[str] = []
    start = 0
    n = len(clean)

    while start < n:
        end = min(start + max_chars, n)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = end - overlap

    return chunks


def safe_str(value: Any) -> str: #values to trimmed string and missing values becomes empty
    if value is None:
        return ""

    s = str(value).strip()
    if s.lower() in {"nan", "none"}:
        return ""

    return s


def build_embedding_text(job: JobTextInput) -> str: # build text block for embeddings
    description_text = html_to_text(job.job_description_html)

    parts = [
        ("Job Title", job.job_title),
        ("Company", job.company_name),
        ("Location", job.location),
        ("Level", job.job_level),
        ("Tags", job.tags),
        ("Description", description_text),
    ]

    lines: list[str] = []
    for label, value in parts:
        value = normalize_whitespace(value)
        if value:
            lines.append(f"{label}: {value}")

    return "\n".join(lines).strip()


def row_to_job_text_input(row: Mapping[str, Any]) -> JobTextInput: #dataset row into jobtext input
    return JobTextInput(
        job_title=safe_str(row[DatasetColumns.JOB_TITLE]),
        company_name=safe_str(row[DatasetColumns.COMPANY_NAME]),
        location=safe_str(row[DatasetColumns.JOB_LOCATION]),
        job_level=safe_str(row[DatasetColumns.JOB_LEVEL]),
        tags=safe_str(row[DatasetColumns.TAGS]),
        job_description_html=safe_str(row[DatasetColumns.JOB_DESCRIPTION]),
    )
