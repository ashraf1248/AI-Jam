import requests

from src.literature_search import (
    OpenAlexLiteratureSearcher,
    build_search_queries,
    reconstruct_openalex_abstract,
)


def test_reconstruct_openalex_abstract_orders_tokens_by_position() -> None:
    abstract = reconstruct_openalex_abstract(
        {
            "cells": [4],
            "respond": [1],
            "to": [2],
            "stress": [3],
            "how": [0],
        }
    )

    assert abstract == "how respond to stress cells"


def test_build_search_queries_compacts_prompt_like_input() -> None:
    queries = build_search_queries(
        query=(
            "Can AI synthesize literature-style evidence and experimental data to identify gaps "
            "and propose testable hypotheses for improving biodegradable starch-PVA polymer films?"
        ),
        domain="Materials Science",
        domain_notes=(
            "Focus on starch-polyvinyl alcohol biodegradable films. Important outcomes are tensile strength, "
            "elongation at break, water uptake, degradation rate, and surface morphology. Variables include "
            "starch ratio, PVA ratio, glycerol concentration, drying temperature, curing humidity, and citric acid crosslinker concentration."
        ),
    )

    assert queries
    assert all(len(query) <= 180 for query in queries)
    assert "starch" in queries[0]
    assert "important outcomes" not in queries[0]


def test_search_retries_with_simpler_query_after_bad_request() -> None:
    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError("bad request", response=self)

        def json(self) -> dict:
            return self._payload

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, str | int]] = []

        def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
            self.calls.append(params)
            if len(self.calls) == 1:
                return FakeResponse(400, {})
            return FakeResponse(
                200,
                {
                    "results": [
                        {
                            "display_name": "Recovered Result",
                            "abstract_inverted_index": {"polymer": [0], "films": [1]},
                            "primary_location": {
                                "source": {"display_name": "Journal"},
                                "landing_page_url": "https://example.org/paper",
                            },
                            "publication_year": 2024,
                            "doi": "10.1000/test",
                            "id": "https://openalex.org/W123",
                            "authorships": [{"author": {"display_name": "A. Author"}}],
                        }
                    ]
                },
            )

    searcher = OpenAlexLiteratureSearcher()
    searcher.session = FakeSession()

    hits = searcher.search(
        query="Very long prompt-like query about biodegradable starch PVA films and many variables",
        max_results=3,
        domain="Materials Science",
        domain_notes="Starch PVA glycerol citric acid crosslinking",
    )

    assert len(hits) == 1
    assert hits[0].title == "Recovered Result"
    assert len(searcher.session.calls) >= 2
    assert len(searcher.session.calls[0]["search"]) <= 180
