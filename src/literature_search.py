from __future__ import annotations

import re
from typing import Any

import requests

from src.schemas import LiteratureSearchHit


_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "data",
    "do",
    "does",
    "evidence",
    "experimental",
    "experiments",
    "for",
    "from",
    "gaps",
    "how",
    "identify",
    "improving",
    "important",
    "include",
    "into",
    "is",
    "it",
    "its",
    "knowledge",
    "literature",
    "materials",
    "mechanism",
    "novel",
    "of",
    "on",
    "or",
    "outcomes",
    "possible",
    "propose",
    "question",
    "research",
    "search",
    "style",
    "synthesize",
    "testable",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "to",
    "variables",
    "was",
    "what",
    "which",
    "with",
}


def reconstruct_openalex_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str:
    if not abstract_inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in abstract_inverted_index.items():
        for index in indexes:
            positions[index] = word
    if not positions:
        return ""
    return " ".join(positions[index] for index in sorted(positions)).strip()


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[\r\n\t]+", " ", value)
    normalized = re.sub(r"[^\w\s\-\/]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_keywords(value: str, limit: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\/]{2,}", _normalize_text(value).lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def build_search_queries(query: str, domain: str = "", domain_notes: str = "") -> list[str]:
    query_terms = _extract_keywords(query, limit=8)
    domain_terms = _extract_keywords(domain, limit=4)
    note_terms = _extract_keywords(domain_notes, limit=8)
    combined_terms = _extract_keywords(" ".join([domain_notes, query, domain]), limit=12)

    candidates = [
        " ".join(note_terms[:6] + query_terms[:4]),
        " ".join(domain_terms + note_terms[:6]),
        " ".join(combined_terms[:10]),
        " ".join(query_terms[:6]),
        " ".join((note_terms or domain_terms)[:6]),
    ]

    unique_queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = re.sub(r"\s+", " ", candidate).strip()[:180].strip()
        if not compact:
            continue
        lowered = compact.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_queries.append(compact)
    return unique_queries or [_normalize_text(query)[:180].strip()]


class OpenAlexLiteratureSearcher:
    def __init__(self, base_url: str = "https://api.openalex.org"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def search(
        self,
        query: str,
        max_results: int = 5,
        domain: str = "",
        domain_notes: str = "",
    ) -> list[LiteratureSearchHit]:
        queries = build_search_queries(query, domain=domain, domain_notes=domain_notes)
        filters = ["has_abstract:true,language:en", "has_abstract:true"]
        last_error: Exception | None = None

        for search_query in queries:
            for filter_value in filters:
                params = {
                    "search": search_query,
                    "per-page": max_results,
                    "filter": filter_value,
                    "sort": "relevance_score:desc",
                }
                try:
                    response = self.session.get(
                        f"{self.base_url}/works",
                        params=params,
                        timeout=30,
                    )
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    last_error = exc
                    if exc.response is not None and exc.response.status_code == 400:
                        continue
                    raise
                payload = response.json()
                hits: list[LiteratureSearchHit] = []
                for item in payload.get("results", []):
                    abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                    if not abstract:
                        continue
                    source = (
                        (item.get("primary_location") or {}).get("source", {}) or {}
                    ).get("display_name", "")
                    authors = [
                        ((authorship.get("author") or {}).get("display_name") or "").strip()
                        for authorship in item.get("authorships", [])
                    ]
                    hits.append(
                        LiteratureSearchHit(
                            title=str(item.get("display_name") or "Untitled paper"),
                            abstract=abstract,
                            source=source or "OpenAlex",
                            publication_year=item.get("publication_year"),
                            doi=str(item.get("doi") or ""),
                            openalex_id=str(item.get("id") or ""),
                            authors=[author for author in authors if author],
                            landing_page_url=str(
                                (item.get("primary_location") or {}).get("landing_page_url") or ""
                            ),
                        )
                    )
                if hits:
                    return hits
        if last_error is not None:
            raise last_error
        return []
