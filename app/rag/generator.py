from __future__ import annotations

import os
from typing import Any

from app.config import Settings


def _fallback_summary(query: str, results: list[dict[str, Any]]) -> str: # Return a simple summary when LLM generation is unavailable
    if not results:
        return (
            f"No strong matches found for '{query}'. "
            "Try adding details like location, experience level, or skills."
        )

    top = results[:3]
    lines = [f"Top matches for '{query}':"]
    for i, job in enumerate(top, start=1):
        title = job.get("job_title", "Unknown title")
        company = job.get("company_name", "Unknown company")
        location = job.get("job_location", "Unknown location")
        level = job.get("job_level", "Unknown level")
        lines.append(f"{i}) {title} — {company} ({location}) [{level}]")

    return "\n".join(lines)


def generate_grounded_answer(
    settings: Settings,
    query: str,
    jobs: list[dict[str, Any]],
) -> str:  # Generate a grounded response using retrieved job chunks only
    if not jobs:
        return f"No relevant jobs found for '{query}'. Try refining your search."

    provider = (settings.llm_provider or "none").strip().lower()
    if provider != "gemini":
        return _fallback_summary(query, jobs)

    # Import Gemini lazily so the app still runs without the dependency
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _fallback_summary(query, jobs)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_summary(query, jobs)

    # Build grounded context strictly from retrieved chunks
    sections: list[str] = []

    for idx, job in enumerate(jobs, start=1):
        header = (
            f"Job {idx}:\n"
            f"Title: {job.get('job_title', '')}\n"
            f"Company: {job.get('company_name', '')}\n"
            f"Location: {job.get('job_location', '')}\n"
            f"Level: {job.get('job_level', '')}\n"
        )

        chunks = job.get("matched_chunks", [])
        if chunks:
            excerpts = "\n".join(f"- {c.get('chunk_text', '')}" for c in chunks)
        else:
            excerpts = "- No relevant excerpts available."

        sections.append(f"{header}\nRelevant excerpts:\n{excerpts}")

    context_text = "\n\n".join(sections)

    prompt = (
        f"User query:\n{query}\n\n"
        "Retrieved job information (use ONLY this data):\n"
        f"{context_text}\n\n"
        "Instructions:\n"
        "- Explain which jobs best match the query and why\n"
        "- Base the answer only on the retrieved excerpts\n"
        "- Do not add external knowledge or assumptions\n"
        "- Keep the response under 120 words"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_output_tokens,
            ),
        )

        text = (response.text or "").strip()
        return text if text else _fallback_summary(query, jobs)

    except Exception:
        return _fallback_summary(query, jobs)
