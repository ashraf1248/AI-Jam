from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.export import export_hypotheses_csv, export_markdown_report, export_result_json
from src.pipeline import CoScientistPipeline, UploadedArtifact
from src.schemas import PipelineInputs, PipelineResult


st.set_page_config(page_title="Hypothesis Forge", layout="wide")


def _uploaded_artifacts(files: list[Any] | None) -> list[UploadedArtifact]:
    artifacts: list[UploadedArtifact] = []
    for file in files or []:
        try:
            artifacts.append(UploadedArtifact(name=file.name, data=file.getvalue()))
        except Exception:
            continue
    return artifacts


def _render_dataset_visuals(result: PipelineResult) -> None:
    if not result.dataset_summaries:
        st.info("No CSV datasets were analyzed.")
        return
    for summary in result.dataset_summaries:
        st.subheader(summary.filename)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Rows", summary.shape[0])
        metric_columns[1].metric("Columns", summary.shape[1])
        metric_columns[2].metric("Numeric Columns", len(summary.numeric_columns))
        metric_columns[3].metric("Categorical Columns", len(summary.categorical_columns))
        st.write(summary.plain_english_summary)
        if summary.strongest_correlations:
            corr_frame = pd.DataFrame([item.model_dump() for item in summary.strongest_correlations])
            st.dataframe(corr_frame, use_container_width=True)
        if summary.descriptive_statistics:
            stats_frame = pd.DataFrame(summary.descriptive_statistics)
            st.dataframe(stats_frame, use_container_width=True)
        if summary.outlier_counts:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(summary.outlier_counts.keys(), summary.outlier_counts.values(), color="#476C9B")
            ax.set_title(f"Possible outliers in {summary.filename}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig, clear_figure=True)


def _render_hypotheses(result: PipelineResult) -> None:
    if not result.hypotheses:
        st.info("No hypotheses generated.")
        return
    for hypothesis in result.hypotheses:
        with st.expander(f"{hypothesis.id} | score={hypothesis.weighted_score:.2f}"):
            st.markdown(f"**Hypothesis**: {hypothesis.hypothesis}")
            st.markdown(f"**Rationale**: {hypothesis.scientific_rationale}")
            st.markdown(f"**Prediction**: {hypothesis.testable_prediction}")
            st.markdown(f"**Expected Result**: {hypothesis.expected_result}")
            st.markdown(f"**Falsification**: {hypothesis.falsification_criteria}")
            st.markdown(f"**Risks / Limitations**: {hypothesis.risks_or_limitations}")
            st.markdown("**Evidence For**")
            st.write(hypothesis.evidence_for)
            st.markdown("**Evidence Against**")
            st.write(hypothesis.evidence_against)
            st.markdown("**Controls**")
            st.write(hypothesis.controls)
            if hypothesis.critique:
                st.markdown("**Critic Scores**")
                st.json(hypothesis.critique.model_dump())


def main() -> None:
    settings = get_settings()
    pipeline = CoScientistPipeline(settings)

    st.title("Hypothesis Forge")
    st.caption(
        "Upload literature, datasets, and images to generate cautious research hypotheses and experiment plans."
    )

    if settings.is_mock_mode:
        st.info("Running in mock/demo mode. Add NVIDIA env vars in `.env` to enable live model calls.")
    else:
        st.success("NVIDIA API configuration detected. Live model endpoints will be used where available.")

    with st.sidebar:
        st.header("Project Setup")
        project_title = st.text_input("Project title", value="Hypothesis Forge Demo")
        research_question = st.text_area(
            "Research question",
            value="What underexplored mechanism could explain the patterns in these uploaded scientific materials?",
            height=140,
        )
        domain = st.selectbox(
            "Domain",
            ["Biology", "Chemistry", "Materials Science", "Physics", "Environmental Science", "Other"],
        )
        domain_notes = st.text_area("Domain notes", height=120)

        st.header("Processing Controls")
        num_hypotheses = st.slider("Number of hypotheses to generate", min_value=1, max_value=10, value=5)
        top_k = st.slider("Number of top hypotheses to keep", min_value=1, max_value=10, value=3)
        include_image_analysis = st.toggle("Include image analysis", value=True)
        run_novelty_check = st.toggle("Run novelty check", value=True)
        run_skeptical_critic = st.toggle("Run skeptical critic", value=True)
        run_pipeline = st.button("Run Co-Scientist Pipeline", use_container_width=True, type="primary")

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Upload Inputs")
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        image_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        notes = st.text_area("Optional text notes", height=180)

    with right:
        st.subheader("Current Session")
        st.write(
            {
                "pdf_count": len(pdf_files or []),
                "csv_count": len(csv_files or []),
                "image_count": len(image_files or []),
                "mock_mode": settings.is_mock_mode,
            }
        )
        if image_files:
            preview_columns = st.columns(min(3, len(image_files)))
            for index, image in enumerate(image_files[:3]):
                preview_columns[index].image(image, caption=image.name, use_container_width=True)

    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None

    if run_pipeline:
        if not research_question.strip():
            st.error("Please add a research question before running the pipeline.")
        else:
            inputs = PipelineInputs(
                project_title=project_title.strip() or "Untitled Hypothesis Forge Project",
                research_question=research_question.strip(),
                domain=domain,
                domain_notes=domain_notes.strip(),
                notes=notes.strip(),
                num_hypotheses=num_hypotheses,
                top_k=min(top_k, num_hypotheses),
                include_image_analysis=include_image_analysis,
                run_novelty_check=run_novelty_check,
                run_skeptical_critic=run_skeptical_critic,
            )
            with st.spinner("Processing uploads and generating research outputs..."):
                try:
                    result = pipeline.run(
                        inputs=inputs,
                        pdfs=_uploaded_artifacts(pdf_files),
                        csvs=_uploaded_artifacts(csv_files),
                        images=_uploaded_artifacts(image_files),
                    )
                    st.session_state["pipeline_result"] = result
                except Exception as exc:  # pragma: no cover - UI safeguard
                    st.error(f"Pipeline failed: {exc}")

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if result:
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        st.subheader("Results Dashboard")
        tabs = st.tabs(
            [
                "Evidence Summary",
                "Data Analysis",
                "Image Observations",
                "Knowledge Gaps",
                "Hypotheses",
                "Experiment Plans",
                "Final Report",
            ]
        )

        with tabs[0]:
            if result.evidence_items:
                evidence_frame = pd.DataFrame([item.model_dump() for item in result.evidence_items])
                st.dataframe(evidence_frame, use_container_width=True)
            else:
                st.info("No PDF evidence was extracted.")

        with tabs[1]:
            _render_dataset_visuals(result)

        with tabs[2]:
            if result.image_observations:
                st.json([item.model_dump() for item in result.image_observations])
            else:
                st.info("No image observations available.")

        with tabs[3]:
            if result.knowledge_gaps:
                for gap in result.knowledge_gaps:
                    st.markdown(f"### {gap.gap_title}")
                    st.write(gap.description)
                    st.write(
                        {
                            "why_it_matters": gap.why_it_matters,
                            "supporting_evidence": gap.supporting_evidence,
                            "missing_information": gap.missing_information,
                            "suggested_next_step": gap.suggested_next_step,
                        }
                    )
            else:
                st.info("No knowledge gaps generated.")

        with tabs[4]:
            _render_hypotheses(result)

        with tabs[5]:
            if result.experiment_plans:
                for plan in result.experiment_plans:
                    st.markdown(f"### {plan.hypothesis_id}")
                    st.json(plan.model_dump())
            else:
                st.info("No experiment plans generated.")

        with tabs[6]:
            st.markdown(result.final_report_markdown)

        st.subheader("Export")
        export_columns = st.columns(3)
        export_columns[0].download_button(
            "Download JSON",
            data=export_result_json(result),
            file_name="hypothesis_forge_result.json",
            mime="application/json",
        )
        export_columns[1].download_button(
            "Download Markdown report",
            data=export_markdown_report(result),
            file_name="hypothesis_forge_report.md",
            mime="text/markdown",
        )
        export_columns[2].download_button(
            "Download CSV of scored hypotheses",
            data=export_hypotheses_csv(result),
            file_name="hypothesis_forge_hypotheses.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
