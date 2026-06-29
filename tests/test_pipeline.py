import io

import pandas as pd
import pytest
from PIL import Image

from src.config import Settings
from src.export import export_hypotheses_csv, export_result_json
from src.pipeline import CoScientistPipeline, UploadedArtifact
from src.retrieval import LocalRetriever, faiss
from src.schemas import LiteratureSearchHit, PipelineInputs, RetrievalDocument


def _png_bytes() -> bytes:
    image = Image.new("RGB", (12, 12), color=(120, 180, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _csv_bytes() -> bytes:
    frame = pd.DataFrame(
        {
            "temperature": [20, 21, 22, 23, 24, 25],
            "yield": [10, 10.4, 10.8, 11.4, 12.0, 12.5],
            "condition": ["A", "A", "B", "B", "B", "A"],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_pipeline_runs_end_to_end_in_mock_mode() -> None:
    pipeline = CoScientistPipeline(
        Settings(
            nvidia_api_key="",
            nvidia_chat_model="",
            nvidia_embed_model="",
            nvidia_vision_model="",
            nvidia_rerank_model="",
        )
    )
    inputs = PipelineInputs(
        project_title="Integration Demo",
        research_question="Which controllable factor could explain the observed pattern?",
        domain="Biology",
        notes="Focus on practical lab follow-up.",
        num_hypotheses=4,
        top_k=2,
    )
    progress_updates: list[tuple[int, str]] = []

    result = pipeline.run(
        inputs=inputs,
        pdfs=[],
        csvs=[UploadedArtifact(name="demo.csv", data=_csv_bytes())],
        images=[UploadedArtifact(name="sample.png", data=_png_bytes())],
        progress_callback=lambda value, message: progress_updates.append((value, message)),
    )

    assert result.mock_mode is True
    assert len(result.dataset_summaries) == 1
    assert len(result.image_observations) == 1
    assert len(result.hypotheses) == 2
    assert len(result.refined_hypotheses) == 2
    assert len(result.experiment_plans) == 2
    assert "# Integration Demo" in result.final_report_markdown
    assert "## Refined Hypotheses" in result.final_report_markdown
    assert result.hypotheses[0].support_traces
    assert result.hypotheses[0].counter_traces
    assert result.refined_hypotheses[0].refined_critique is not None
    assert result.refined_hypotheses[0].score_delta_vs_original is not None
    assert result.refined_hypotheses[0].support_traces
    assert progress_updates[0][0] == 5
    assert progress_updates[-1][0] == 100

    exported_json = export_result_json(result).decode("utf-8")
    exported_csv = export_hypotheses_csv(result).decode("utf-8")
    assert '"mock_mode": true' in exported_json
    assert "id,hypothesis,weighted_score,confidence_score,should_keep,main_weakness" in exported_csv


def test_pipeline_skips_refinement_when_disabled() -> None:
    pipeline = CoScientistPipeline(
        Settings(
            nvidia_api_key="",
            nvidia_chat_model="",
            nvidia_embed_model="",
            nvidia_vision_model="",
            nvidia_rerank_model="",
        )
    )
    inputs = PipelineInputs(
        project_title="No Refinement",
        research_question="Can we keep the original hypotheses only?",
        domain="Biology",
        num_hypotheses=3,
        top_k=2,
        run_hypothesis_refinement=False,
    )

    result = pipeline.run(
        inputs=inputs,
        pdfs=[],
        csvs=[UploadedArtifact(name="demo.csv", data=_csv_bytes())],
        images=[],
    )

    assert result.refined_hypotheses == []
    assert "No refined hypotheses were generated." in result.final_report_markdown


def test_pipeline_uses_mocked_external_literature_hits() -> None:
    pipeline = CoScientistPipeline(
        Settings(
            nvidia_api_key="",
            nvidia_chat_model="",
            nvidia_embed_model="",
            nvidia_vision_model="",
            nvidia_rerank_model="",
        )
    )
    pipeline._search_literature = lambda inputs, warnings: [
        LiteratureSearchHit(
            title="External Mechanism Paper",
            abstract="A stress-responsive pathway changes when moderator X is present in condition Y.",
            source="Journal of Testing",
            publication_year=2024,
            doi="10.1000/example",
            openalex_id="https://openalex.org/W123",
            authors=["A. Researcher"],
            landing_page_url="https://example.org/paper",
        )
    ]
    inputs = PipelineInputs(
        project_title="External Search Demo",
        research_question="Which mechanism may explain the pattern?",
        domain="Biology",
        num_hypotheses=3,
        top_k=2,
        run_literature_search=True,
        max_literature_results=3,
    )

    result = pipeline.run(
        inputs=inputs,
        pdfs=[],
        csvs=[UploadedArtifact(name="demo.csv", data=_csv_bytes())],
        images=[],
    )

    assert len(result.literature_hits) == 1
    assert result.inputs_analyzed["literature_search"][0] == "External Mechanism Paper (2024)"
    assert any(trace.source_type == "literature_search" for trace in result.hypotheses[0].support_traces)
    assert "## External Literature Search" in result.final_report_markdown


def test_pipeline_uses_real_external_search_even_in_mock_mode() -> None:
    pipeline = CoScientistPipeline(
        Settings(
            nvidia_api_key="",
            nvidia_chat_model="",
            nvidia_embed_model="",
            nvidia_vision_model="",
            nvidia_rerank_model="",
        )
    )
    pipeline.literature_searcher.search = lambda **_: [
        LiteratureSearchHit(
            title="Real External Paper",
            abstract="A real abstract from external search.",
            source="OpenAlex Test Source",
            publication_year=2022,
            doi="10.2000/real",
            openalex_id="https://openalex.org/W999",
            authors=["B. Scientist"],
            landing_page_url="https://example.org/real-paper",
        )
    ]
    inputs = PipelineInputs(
        project_title="Mock Search Demo",
        research_question="Which hidden variable could explain the phenomenon?",
        domain="Biology",
        run_literature_search=True,
        max_literature_results=2,
    )

    result = pipeline.run(
        inputs=inputs,
        pdfs=[],
        csvs=[],
        images=[],
    )

    assert len(result.literature_hits) == 1
    assert result.literature_hits[0].title == "Real External Paper"
    assert all("fallback" not in warning.lower() for warning in result.warnings)


def test_pipeline_generates_demo_literature_hits_when_search_fails() -> None:
    pipeline = CoScientistPipeline(
        Settings(
            nvidia_api_key="",
            nvidia_chat_model="",
            nvidia_embed_model="",
            nvidia_vision_model="",
            nvidia_rerank_model="",
        )
    )
    pipeline.literature_searcher.search = lambda **_: (_ for _ in ()).throw(RuntimeError("network down"))
    inputs = PipelineInputs(
        project_title="Fallback Search Demo",
        research_question="Which hidden variable could explain the phenomenon?",
        domain="Biology",
        run_literature_search=True,
        max_literature_results=2,
    )

    result = pipeline.run(
        inputs=inputs,
        pdfs=[],
        csvs=[],
        images=[],
    )

    assert len(result.literature_hits) == 2
    assert result.literature_hits[0].source == "Mock Literature Search"
    assert any("fallback papers were used" in warning.lower() for warning in result.warnings)


@pytest.mark.skipif(faiss is None, reason="faiss is not available in this environment")
def test_vector_retrieval_uses_query_embedder() -> None:
    retriever = LocalRetriever(
        query_embedder=lambda text: [1.0, 0.0] if text == "alpha query" else [0.0, 1.0]
    )
    retriever.add_documents(
        [
            RetrievalDocument(text="document one", source="a"),
            RetrievalDocument(text="document two", source="b"),
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    results = retriever.query("alpha query", top_k=1)

    assert len(results) == 1
    assert results[0].source == "a"
