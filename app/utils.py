from typing import Dict, Any


def sanitize_job_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_title": meta.get("job_title"),
        "company_name": meta.get("company_name"),
        "job_location": meta.get("job_location"),
        "job_level": meta.get("job_level"),
        "publication_date": meta.get("publication_date"),
        "job_category": meta.get("job_category"),
    }
