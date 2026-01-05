from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env
load_dotenv()


@dataclass(frozen=True)
class Settings:
    dataset_path: str
    faiss_index_path: str
    metadata_path: str
    embedding_model: str
    top_k: int

    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_output_tokens: int


def _require_env(name: str) -> str: #read env variable or raise error
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {value!r})") from exc


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float (got {value!r})") from exc


def get_settings() -> Settings:
    dataset_path = _require_env("DATASET_PATH")
    faiss_index_path = _require_env("FAISS_INDEX_PATH")
    metadata_path = _require_env("METADATA_PATH")
    embedding_model = _require_env("EMBEDDING_MODEL")

    top_k = _parse_int("TOP_K", os.getenv("TOP_K", "5"))

    llm_provider = os.getenv("LLM_PROVIDER", "none")
    llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    llm_temperature = _parse_float("LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_output_tokens = _parse_int(
        "LLM_MAX_OUTPUT_TOKENS", os.getenv("LLM_MAX_OUTPUT_TOKENS", "250")
    )

    return Settings(
        dataset_path=dataset_path,
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
        top_k=top_k,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_output_tokens=llm_max_output_tokens,
    )
