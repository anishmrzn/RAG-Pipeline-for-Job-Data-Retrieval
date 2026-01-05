from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings


@dataclass(frozen=True) 
class RetrievedJob: # ranked job result aggregated from one or more matching chunks
    score: float
    metadata: dict[str, Any]
    matched_chunks: list[dict[str, Any]]


class JobRetriever: # Retrieve top jobs using a FAISS vector index with a light keyword boost

    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "to", "for", "in", "on", "with", "of", "at",
        "as", "by", "from", "is", "are", "be", "this", "that", "it", "you", "we",
        "job", "role", "position",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(settings.embedding_model)
        self._index = self._load_index(settings.faiss_index_path)
        self._metadata = self._load_metadata(settings.metadata_path)

    @staticmethod
    def _tokenize(text: str) -> list[str]: # Lowercase alphanumeric tokenization with simple stopword removal
        tokens = JobRetriever._TOKEN_RE.findall(text.lower())
        return [t for t in tokens if t not in JobRetriever._STOPWORDS and len(t) > 1]

    @staticmethod
    def _job_key(meta: dict[str, Any]) -> str: # Stable job identifier for grouping chunks into a single job listing
        job_id = str(meta.get("id", "")).strip()
        return job_id if job_id else f"row_{meta.get('row_id', '')}"

    @staticmethod
    def _keyword_boost(query: str, meta: dict[str, Any]) -> float: # Small hybrid boost added to the vector score
        q = query.lower()

        title = str(meta.get("job_title", ""))
        category = str(meta.get("job_category", ""))
        tags = str(meta.get("tags", ""))
        location = str(meta.get("job_location", ""))
        level = str(meta.get("job_level", ""))
        preview = str(meta.get("text_preview", ""))

        level_lower = level.lower()
        location_lower = location.lower()

        boost = 0.0

        if any(x in q for x in ("intern", "internship")):
            if "intern" in level_lower:
                boost += 0.18
            elif "senior" in level_lower:
                boost -= 0.12
            else:
                boost -= 0.04

        if "senior" in q:
            if "senior" in level_lower:
                boost += 0.12
            elif "intern" in level_lower:
                boost -= 0.06

        if any(x in q for x in ("beginner", "entry", "entry-level", "junior", "jr", "trainee", "fresher")):
            if any(x in level_lower for x in ("entry", "junior", "intern", "trainee", "fresher")):
                boost += 0.14
            elif any(x in level_lower for x in ("mid", "associate")):
                boost -= 0.04
            elif any(x in level_lower for x in ("senior", "staff", "lead", "manager", "principal", "director")):
                boost -= 0.18

        if any(x in q for x in ("remote", "work from home", "wfh")) and "remote" in location_lower:
            boost += 0.10

        if "hybrid" in q and "hybrid" in location_lower:
            boost += 0.08

        q_tokens = set(JobRetriever._tokenize(q))
        if not q_tokens:
            return boost

        fields: list[tuple[str, float]] = [
            (title, 0.05),
            (tags, 0.04),
            (category, 0.03),
            (location, 0.02),
            (preview, 0.02),
        ]

        for text, weight in fields:
            overlap = q_tokens.intersection(JobRetriever._tokenize(text))
            boost += weight * len(overlap)

        return boost

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedJob]: # Retrieve top job listings for a query
        q = query.strip()
        if not q:
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else self._settings.top_k

        query_vec = self._embed_query(q)

        candidate_k = min(max(k * 30, 50), 200)
        scores, indices = self._search(query_vec, candidate_k)

        grouped: dict[str, dict[str, Any]] = {}

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._metadata):
                continue

            chunk_meta = self._metadata[idx]
            boosted = float(score) + self._keyword_boost(q, chunk_meta)
            key = self._job_key(chunk_meta)

            if key not in grouped:
                grouped[key] = {
                    "best_score": boosted,
                    "job_meta": chunk_meta,
                    "chunks": [(boosted, chunk_meta)],
                }
                continue

            grouped[key]["chunks"].append((boosted, chunk_meta))
            if boosted > grouped[key]["best_score"]:
                grouped[key]["best_score"] = boosted
                grouped[key]["job_meta"] = chunk_meta

        results: list[RetrievedJob] = []
        for item in grouped.values():
            chunks_sorted = sorted(item["chunks"], key=lambda x: x[0], reverse=True)
            top_chunks = [m for _, m in chunks_sorted[:2]]

            results.append(
                RetrievedJob(
                    score=float(item["best_score"]),
                    metadata=item["job_meta"],
                    matched_chunks=top_chunks,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _embed_query(self, query: str) -> np.ndarray: # Embed and L2-normalize the query
        vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def _search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: # Run FAISS search and return (scores, indices)
        scores, indices = self._index.search(query_vec, top_k)
        return scores[0], indices[0]

    @staticmethod
    def _load_index(index_path: str) -> faiss.Index:
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        return faiss.read_index(str(path))

    @staticmethod
    def _load_metadata(metadata_path: str) -> list[dict[str, Any]]:
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Metadata file must contain a JSON list.")

        return data
