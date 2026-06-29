from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.export import export_hypotheses_csv, export_markdown_report, export_result_json
from src.pipeline import CoScientistPipeline, UploadedArtifact
from src.project_store import ProjectStore
from src.schemas import EvidenceTrace, PipelineInputs, PipelineResult, SavedProject, SavedProjectSummary


st.set_page_config(page_title="Hypothesis Forge", layout="wide")

MAX_PDF_MB = 25
MAX_CSV_MB = 10
MAX_IMAGE_MB = 10

DEFAULT_SESSION_VALUES = {
    "project_title": "Hypothesis Forge Demo",
    "research_question": "What underexplored mechanism could explain the patterns in these uploaded scientific materials?",
    "domain": "Biology",
    "domain_notes": "",
    "notes": "",
    "num_hypotheses": 5,
    "top_k": 3,
    "include_image_analysis": True,
    "run_literature_search": False,
    "max_literature_results": 5,
    "run_novelty_check": True,
    "run_skeptical_critic": True,
    "run_hypothesis_refinement": True,
    "pipeline_result": None,
    "active_project_id": None,
    "save_status_message": "",
    "uploader_version": 0,
}


def _empty_artifact_map() -> dict[str, list[UploadedArtifact]]:
    return {"pdfs": [], "csvs": [], "images": []}


def _initialize_session_state() -> None:
    for key, value in DEFAULT_SESSION_VALUES.items():
        st.session_state.setdefault(key, value)
    if "stored_project_artifacts" not in st.session_state:
        st.session_state["stored_project_artifacts"] = _empty_artifact_map()
    if "stored_upload_manifest" not in st.session_state:
        st.session_state["stored_upload_manifest"] = {"pdfs": [], "csvs": [], "images": []}
    if "selected_saved_project_label" not in st.session_state:
        st.session_state["selected_saved_project_label"] = None


def _reset_project_state() -> None:
    next_uploader_version = st.session_state.get("uploader_version", 0) + 1
    for key, value in DEFAULT_SESSION_VALUES.items():
        st.session_state[key] = value
    st.session_state["stored_project_artifacts"] = _empty_artifact_map()
    st.session_state["stored_upload_manifest"] = {"pdfs": [], "csvs": [], "images": []}
    st.session_state["uploader_version"] = next_uploader_version


def _build_inputs_from_state() -> PipelineInputs:
    return PipelineInputs(
        project_title=st.session_state["project_title"].strip() or "Untitled Hypothesis Forge Project",
        research_question=st.session_state["research_question"].strip(),
        domain=st.session_state["domain"],
        domain_notes=st.session_state["domain_notes"].strip(),
        notes=st.session_state["notes"].strip(),
        num_hypotheses=st.session_state["num_hypotheses"],
        top_k=min(st.session_state["top_k"], st.session_state["num_hypotheses"]),
        include_image_analysis=st.session_state["include_image_analysis"],
        run_literature_search=st.session_state["run_literature_search"],
        max_literature_results=st.session_state["max_literature_results"],
        run_novelty_check=st.session_state["run_novelty_check"],
        run_skeptical_critic=st.session_state["run_skeptical_critic"],
        run_hypothesis_refinement=st.session_state["run_hypothesis_refinement"],
    )


def _apply_saved_project(project: SavedProject) -> None:
    st.session_state["project_title"] = project.inputs.project_title
    st.session_state["research_question"] = project.inputs.research_question
    st.session_state["domain"] = project.inputs.domain
    st.session_state["domain_notes"] = project.inputs.domain_notes
    st.session_state["notes"] = project.inputs.notes
    st.session_state["num_hypotheses"] = project.inputs.num_hypotheses
    st.session_state["top_k"] = project.inputs.top_k
    st.session_state["include_image_analysis"] = project.inputs.include_image_analysis
    st.session_state["run_literature_search"] = project.inputs.run_literature_search
    st.session_state["max_literature_results"] = project.inputs.max_literature_results
    st.session_state["run_novelty_check"] = project.inputs.run_novelty_check
    st.session_state["run_skeptical_critic"] = project.inputs.run_skeptical_critic
    st.session_state["run_hypothesis_refinement"] = project.inputs.run_hypothesis_refinement
    st.session_state["pipeline_result"] = project.result
    st.session_state["active_project_id"] = project.project_id
    st.session_state["save_status_message"] = (
        f"Loaded saved project '{project.project_title}' from {project.updated_at}."
    )
    st.session_state["stored_project_artifacts"] = ProjectStore.deserialize_artifacts(
        project.stored_artifacts
    )
    st.session_state["stored_upload_manifest"] = project.upload_manifest
    st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1


def _format_saved_project(summary: SavedProjectSummary) -> str:
    suffix = "results saved" if summary.has_result else "draft only"
    return f"{summary.project_title} | {summary.updated_at} | {suffix}"


def _resolve_artifacts(
    current_artifacts: dict[str, list[UploadedArtifact]],
    stored_artifacts: dict[str, list[UploadedArtifact]],
) -> dict[str, list[UploadedArtifact]]:
    resolved: dict[str, list[UploadedArtifact]] = {}
    for artifact_type in ("pdfs", "csvs", "images"):
        resolved[artifact_type] = current_artifacts.get(artifact_type) or stored_artifacts.get(
            artifact_type,
            [],
        )
    return resolved


def _save_project_snapshot(
    store: ProjectStore,
    artifacts_by_type: dict[str, list[UploadedArtifact]],
) -> SavedProject:
    snapshot = store.save_project(
        inputs=_build_inputs_from_state(),
        result=st.session_state.get("pipeline_result"),
        artifacts_by_type=artifacts_by_type,
        project_id=st.session_state.get("active_project_id"),
    )
    st.session_state["active_project_id"] = snapshot.project_id
    st.session_state["save_status_message"] = (
        f"Saved '{snapshot.project_title}' at {snapshot.updated_at}."
    )
    st.session_state["stored_project_artifacts"] = ProjectStore.deserialize_artifacts(
        snapshot.stored_artifacts
    )
    st.session_state["stored_upload_manifest"] = snapshot.upload_manifest
    return snapshot


def _uploaded_artifacts(
    files: list[Any] | None,
    label: str,
    max_size_mb: int,
) -> tuple[list[UploadedArtifact], list[str]]:
    artifacts: list[UploadedArtifact] = []
    warnings: list[str] = []
    max_size_bytes = max_size_mb * 1024 * 1024
    for file in files or []:
        size_bytes = int(getattr(file, "size", 0) or 0)
        if size_bytes > max_size_bytes:
            warnings.append(
                f"{label} '{file.name}' was skipped because it is larger than {max_size_mb} MB."
            )
            continue
        try:
            data = file.getvalue()
        except Exception as exc:
            warnings.append(f"{label} '{getattr(file, 'name', 'unknown')}' could not be read: {exc}")
            continue
        if not data:
            warnings.append(f"{label} '{file.name}' was empty and was skipped.")
            continue
        if len(data) > max_size_bytes:
            warnings.append(
                f"{label} '{file.name}' was skipped because it is larger than {max_size_mb} MB."
            )
            continue
        artifacts.append(UploadedArtifact(name=file.name, data=data))
    return artifacts, warnings


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
            if hypothesis.support_traces:
                st.markdown("**Supporting Source Traces**")
                _render_trace_table(hypothesis.support_traces)
            if hypothesis.counter_traces:
                st.markdown("**Counter-Evidence Traces**")
                _render_trace_table(hypothesis.counter_traces)


def _render_trace_table(traces: list[EvidenceTrace]) -> None:
    frame = pd.DataFrame([trace.model_dump() for trace in traces])
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_refined_hypotheses(result: PipelineResult) -> None:
    if not result.refined_hypotheses:
        st.info("No refined hypotheses generated.")
        return
    for refined in result.refined_hypotheses:
        with st.expander(refined.hypothesis_id):
            st.markdown(f"**Original Hypothesis**: {refined.original_hypothesis}")
            st.markdown(f"**Improved Hypothesis**: {refined.improved_hypothesis}")
            score_columns = st.columns(3)
            score_columns[0].metric("Refined Score", f"{(refined.weighted_score or 0.0):.2f}")
            score_columns[1].metric(
                "Delta vs Original",
                f"{(refined.score_delta_vs_original or 0.0):+.2f}",
            )
            score_columns[2].metric(
                "Should Keep",
                "Yes" if refined.refined_critique and refined.refined_critique.should_keep else "Review",
            )
            st.markdown(
                f"**Why Original Was Insufficient**: {refined.why_original_was_insufficient}"
            )
            st.markdown(f"**Hidden Assumption**: {refined.hidden_assumption}")
            st.markdown(f"**Sharper Mechanism**: {refined.sharper_mechanism}")
            st.markdown(
                f"**Key Interaction / Missing Variable**: {refined.key_interaction_or_missing_variable}"
            )
            st.markdown(f"**Revised Prediction**: {refined.revised_prediction}")
            st.markdown(
                f"**Mechanism-Discriminating Experiment**: {refined.mechanism_discriminating_experiment}"
            )
            st.markdown(
                "**What Result Would Distinguish Mechanisms**: "
                f"{refined.what_result_would_distinguish_mechanisms}"
            )
            st.markdown(f"**Why This Is More Novel**: {refined.why_this_is_more_novel}")
            st.markdown(f"**Residual Uncertainty**: {refined.residual_uncertainty}")
            if refined.refined_critique:
                st.markdown("**Re-Critic Scores**")
                st.json(refined.refined_critique.model_dump())
            if refined.support_traces:
                st.markdown("**Supporting Source Traces**")
                _render_trace_table(refined.support_traces)
            if refined.counter_traces:
                st.markdown("**Counter-Evidence Traces**")
                _render_trace_table(refined.counter_traces)


def _render_literature_hits(result: PipelineResult) -> None:
    if not result.literature_hits:
        st.info("No external literature hits were analyzed.")
        return
    frame = pd.DataFrame([hit.model_dump() for hit in result.literature_hits])
    st.dataframe(frame, use_container_width=True, hide_index=True)


def main() -> None:
    settings = get_settings()
    pipeline = CoScientistPipeline(settings)
    store = ProjectStore()
    _initialize_session_state()

    saved_projects = store.list_projects()
    saved_project_lookup = {_format_saved_project(project): project.project_id for project in saved_projects}

    st.title("Hypothesis Forge")
    st.caption(
        "Upload literature, datasets, and images to generate cautious research hypotheses and experiment plans."
    )

    if settings.is_mock_mode:
        st.info("Running in mock/demo mode. Add NVIDIA env vars in `.env` to enable live model calls.")
    else:
        st.success("NVIDIA API configuration detected. Live model endpoints will be used where available.")

    with st.sidebar:
        st.header("Saved Projects")
        if saved_project_lookup:
            labels = list(saved_project_lookup.keys())
            if st.session_state["selected_saved_project_label"] not in labels:
                st.session_state["selected_saved_project_label"] = labels[0]
            st.selectbox(
                "Open a saved project",
                options=labels,
                key="selected_saved_project_label",
            )
            load_saved_project = st.button("Load Saved Project", use_container_width=True)
        else:
            st.caption("No saved projects yet. Save a draft or run the pipeline to create one.")
            load_saved_project = False
        start_new_project = st.button("Start New Project", use_container_width=True)

    if start_new_project:
        _reset_project_state()
        st.rerun()

    if load_saved_project:
        selected_label = st.session_state.get("selected_saved_project_label")
        selected_id = saved_project_lookup.get(selected_label)
        project = store.load_project(selected_id)
        if project is not None:
            _apply_saved_project(project)
            st.rerun()
        st.warning("That saved project could not be loaded.")

    with st.sidebar:
        st.header("Project Setup")
        st.text_input("Project title", key="project_title")
        st.text_area("Research question", height=140, key="research_question")
        st.selectbox(
            "Domain",
            ["Biology", "Chemistry", "Materials Science", "Physics", "Environmental Science", "Other"],
            key="domain",
        )
        st.text_area("Domain notes", height=120, key="domain_notes")

        st.header("Processing Controls")
        st.slider("Number of hypotheses to generate", min_value=1, max_value=10, key="num_hypotheses")
        st.slider("Number of top hypotheses to keep", min_value=1, max_value=10, key="top_k")
        st.toggle("Include image analysis", key="include_image_analysis")
        st.toggle("Run external literature search", key="run_literature_search")
        st.slider("Max external papers", min_value=1, max_value=10, key="max_literature_results")
        st.toggle("Run novelty check", key="run_novelty_check")
        st.toggle("Run skeptical critic", key="run_skeptical_critic")
        st.toggle("Run hypothesis refinement", key="run_hypothesis_refinement")
        save_project = st.button("Save Current Project", use_container_width=True)
        run_pipeline = st.button("Run Co-Scientist Pipeline", use_container_width=True, type="primary")
        if st.session_state.get("active_project_id"):
            st.caption(f"Active project ID: `{st.session_state['active_project_id']}`")

    left, right = st.columns([1.2, 1.0])
    uploader_version = st.session_state.get("uploader_version", 0)
    with left:
        st.subheader("Upload Inputs")
        st.caption(
            f"Recommended size limits: PDFs up to {MAX_PDF_MB} MB, CSVs up to {MAX_CSV_MB} MB, "
            f"images up to {MAX_IMAGE_MB} MB each."
        )
        pdf_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_files_{uploader_version}",
        )
        csv_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key=f"csv_files_{uploader_version}",
        )
        image_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"image_files_{uploader_version}",
        )
        st.text_area("Optional text notes", height=180, key="notes")
        if any(st.session_state.get("stored_upload_manifest", {}).values()):
            st.info(
                "This project has saved files on disk. If you rerun without uploading replacements, "
                "the app can reuse the saved PDFs, CSVs, and images."
            )

    with right:
        st.subheader("Current Session")
        stored_artifacts = st.session_state.get("stored_project_artifacts", _empty_artifact_map())
        st.write(
            {
                "pdf_count": len(pdf_files or []),
                "csv_count": len(csv_files or []),
                "image_count": len(image_files or []),
                "saved_pdf_count": len(stored_artifacts.get("pdfs", [])),
                "saved_csv_count": len(stored_artifacts.get("csvs", [])),
                "saved_image_count": len(stored_artifacts.get("images", [])),
                "external_literature_search": st.session_state["run_literature_search"],
                "mock_mode": settings.is_mock_mode,
            }
        )
        if image_files:
            preview_columns = st.columns(min(3, len(image_files)))
            for index, image in enumerate(image_files[:3]):
                preview_columns[index].image(image, caption=image.name, use_container_width=True)
        elif stored_artifacts.get("images"):
            st.caption(
                "Saved images are available for this project, even though they are not shown in the uploader."
            )

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    current_pdf_artifacts, pdf_warnings = _uploaded_artifacts(pdf_files, "PDF", MAX_PDF_MB)
    current_csv_artifacts, csv_warnings = _uploaded_artifacts(csv_files, "CSV", MAX_CSV_MB)
    current_image_artifacts, image_warnings = _uploaded_artifacts(image_files, "Image", MAX_IMAGE_MB)
    current_artifacts = {
        "pdfs": current_pdf_artifacts,
        "csvs": current_csv_artifacts,
        "images": current_image_artifacts,
    }
    stored_artifacts = st.session_state.get("stored_project_artifacts", _empty_artifact_map())
    resolved_artifacts = _resolve_artifacts(current_artifacts, stored_artifacts)
    upload_warnings = pdf_warnings + csv_warnings + image_warnings

    if save_project:
        if not st.session_state["research_question"].strip():
            st.error("Please add a research question before saving the project.")
        else:
            snapshot = _save_project_snapshot(store, resolved_artifacts)
            st.success(
                f"Saved project '{snapshot.project_title}'. You can reopen it later from Saved Projects."
            )

    if run_pipeline:
        if not st.session_state["research_question"].strip():
            st.error("Please add a research question before running the pipeline.")
        else:
            if not (
                resolved_artifacts["pdfs"]
                or resolved_artifacts["csvs"]
                or resolved_artifacts["images"]
                or st.session_state["notes"].strip()
                or st.session_state["run_literature_search"]
            ):
                st.error(
                    "Add at least one PDF, CSV, image, or note, or enable external literature search before running the pipeline."
                )
            else:
                inputs = _build_inputs_from_state()

                def _progress_update(value: int, message: str) -> None:
                    progress_placeholder.progress(value)
                    status_placeholder.info(message)

                with st.spinner("Processing uploads and generating research outputs..."):
                    try:
                        for warning in upload_warnings:
                            st.warning(warning)
                        reused_sources = [
                            label
                            for artifact_type, label in (
                                ("pdfs", "PDFs"),
                                ("csvs", "CSVs"),
                                ("images", "images"),
                            )
                            if not current_artifacts[artifact_type]
                            and resolved_artifacts[artifact_type]
                        ]
                        if reused_sources:
                            st.info(
                                "Reusing saved files for: " + ", ".join(reused_sources) + "."
                            )
                        result = pipeline.run(
                            inputs=inputs,
                            pdfs=resolved_artifacts["pdfs"],
                            csvs=resolved_artifacts["csvs"],
                            images=resolved_artifacts["images"],
                            progress_callback=_progress_update,
                        )
                        st.session_state["pipeline_result"] = result
                        snapshot = _save_project_snapshot(store, resolved_artifacts)
                        status_placeholder.success(
                            f"Pipeline finished successfully and saved to '{snapshot.project_title}'."
                        )
                    except Exception as exc:  # pragma: no cover - UI safeguard
                        progress_placeholder.empty()
                        status_placeholder.error("The pipeline stopped before finishing.")
                        st.error(f"Pipeline failed: {exc}")

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if st.session_state.get("save_status_message"):
        st.caption(st.session_state["save_status_message"])
    if result:
        if st.session_state.get("run_literature_search") and not result.literature_hits:
            st.info(
                "External literature search is enabled, but this result has no search hits. "
                "If this is an older saved project, rerun the pipeline. Otherwise, check the warnings for search issues."
            )
        if st.session_state.get("run_hypothesis_refinement") and not result.refined_hypotheses:
            st.info(
                "Hypothesis refinement is enabled, but this result has no refined hypotheses. "
                "If this is an older saved project, rerun the pipeline to generate them."
            )
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        st.subheader("Results Dashboard")
        tabs = st.tabs(
            [
                "Literature Search",
                "Evidence Summary",
                "Data Analysis",
                "Image Observations",
                "Knowledge Gaps",
                "Hypotheses",
                "Refined Hypotheses",
                "Experiment Plans",
                "Final Report",
            ]
        )

        with tabs[0]:
            _render_literature_hits(result)

        with tabs[1]:
            if result.evidence_items:
                evidence_frame = pd.DataFrame([item.model_dump() for item in result.evidence_items])
                st.dataframe(evidence_frame, use_container_width=True)
            else:
                st.info("No literature evidence was extracted from uploaded PDFs or external search results.")

        with tabs[2]:
            _render_dataset_visuals(result)

        with tabs[3]:
            if result.image_observations:
                st.json([item.model_dump() for item in result.image_observations])
            else:
                st.info("No image observations available.")

        with tabs[4]:
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

        with tabs[5]:
            _render_hypotheses(result)

        with tabs[6]:
            _render_refined_hypotheses(result)

        with tabs[7]:
            if result.experiment_plans:
                for plan in result.experiment_plans:
                    st.markdown(f"### {plan.hypothesis_id}")
                    st.json(plan.model_dump())
            else:
                st.info("No experiment plans generated.")

        with tabs[8]:
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
