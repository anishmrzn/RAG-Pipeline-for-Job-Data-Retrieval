from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.constants import DatasetColumns
from app.rag.preprocess import build_embedding_text, chunk_text, row_to_job_text_input


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def load_jobs(dataset_path: str) -> pd.DataFrame: #load csv into dataframe
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset was loaded, but it contains no rows.")

    return df


def build_texts_and_metadata(df: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]]]: #convert job row into chunked text
    texts: list[str] = []
    metadata_list: list[dict[str, Any]] = []

    records = df.to_dict(orient="records")

    for row_index, row in enumerate(records):
        job_input = row_to_job_text_input(row)
        full_text = build_embedding_text(job_input)

        chunks = chunk_text(full_text, max_chars=900, overlap=150)
        if not chunks:
            continue

        for chunk_index, chunk in enumerate(chunks):
            meta: dict[str, Any] = {
                "row_id": row_index,
                "id": row.get(DatasetColumns.ID, ""),
                "job_category": row.get(DatasetColumns.JOB_CATEGORY, ""),
                "publication_date": row.get(DatasetColumns.PUBLICATION_DATE, ""),
                "job_title": job_input.job_title,
                "company_name": job_input.company_name,
                "job_location": job_input.location,
                "job_level": job_input.job_level,
                "tags": job_input.tags,
                "chunk_id": chunk_index,
                "chunk_text": chunk,
                "text_preview": chunk[:400],
            }

            texts.append(chunk)
            metadata_list.append(meta)

    return texts, metadata_list


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray: #Normalize embeddings to unit length for cosine similarity
    emb = embeddings.astype("float32", copy=False)
    faiss.normalize_L2(emb)
    return emb


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index: #Create a FAISS index using inner product on normalized embeddings
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array with shape (n, dim).")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_metadata(metadata_path: str, metadata_list: list[dict[str, Any]]) -> None: # Save chunk-level metadata as JSON
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


def main() -> None:
    settings = get_settings()

    print("Reading the jobs dataset...")
    df = load_jobs(settings.dataset_path)

    print("Preparing text chunks and metadata...")
    texts, metadata_list = build_texts_and_metadata(df)
    if not texts:
        raise ValueError("No text chunks were produced. Check the dataset content and preprocessing.")

    print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = normalize_embeddings(embeddings)

    print("Building FAISS index...")
    index = create_faiss_index(embeddings)

    print(f"Saving FAISS index to: {settings.faiss_index_path}")
    ensure_parent_dir(settings.faiss_index_path)
    faiss.write_index(index, settings.faiss_index_path)

    print(f"Saving metadata to: {settings.metadata_path}")
    save_metadata(settings.metadata_path, metadata_list)

    print("Index build complete.")
    print(f"Chunks indexed: {len(metadata_list)}")
    print(f"Index file: {settings.faiss_index_path}")
    print(f"Metadata file: {settings.metadata_path}")


if __name__ == "__main__":
    main()
