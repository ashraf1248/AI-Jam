from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from pydantic import ValidationError

from src import agents
from src.config import Settings
from src.data_processor import analyze_csv_bytes
from src.image_processor import fallback_image_observation
from src.literature_search import OpenAlexLiteratureSearcher
from src.nvidia_client import NvidiaClient
from src.pdf_processor import extract_chunks_from_pdf
from src.retrieval import LocalRetriever, _mock_embedding
from src.schemas import (
    DatasetSummary,
    EvidenceTrace,
    EvidenceItem,
    ExperimentPlan,
    FinalReportSections,
    Hypothesis,
    HypothesisCritique,
    ImageObservation,
    KnowledgeGap,
    LiteratureSearchHit,
    PipelineInputs,
    PipelineResult,
    RefinedHypothesis,
    RetrievalDocument,
)
from src.scoring import rank_hypotheses, score_refined_hypotheses
from src.utils import parse_json_with_repair, preview_list, stable_seed


@dataclass(slots=True)
class UploadedArtifact:
    name: str
    data: bytes


ProgressCallback = Callable[[int, str], None]


def _keyword_overlap_score(left: str, right: str) -> int:
    tokens_left = {token for token in re.findall(r"[a-zA-Z0-9_]+", left.lower()) if len(token) > 3}
    tokens_right = {token for token in re.findall(r"[a-zA-Z0-9_]+", right.lower()) if len(token) > 3}
    return len(tokens_left & tokens_right)


def _mock_evidence_from_chunk(chunk: str, source_filename: str, index: int) -> list[EvidenceItem]:
    snippet = chunk[:220].strip()
    if not snippet:
        return []
    return [
        EvidenceItem(
            claim=f"Mock evidence {index + 1}: {snippet[:120]}",
            mechanism="Potential mechanism requires validation against broader literature.",
            variables=preview_list([word.strip(".,") for word in snippet.split() if len(word) > 6], limit=3),
            method="PDF chunk extraction in demo mode.",
            limitation="Generated without a live language model; may miss nuance.",
            citation=source_filename,
            confidence=0.45,
        )
    ]


def _literature_hit_label(hit: LiteratureSearchHit) -> str:
    if hit.publication_year:
        return f"{hit.title} ({hit.publication_year})"
    return hit.title


def _mock_literature_hits(inputs: PipelineInputs) -> list[LiteratureSearchHit]:
    domain = inputs.domain_notes.strip() or inputs.domain
    base_title = inputs.research_question.strip().rstrip("?")
    return [
        LiteratureSearchHit(
            title=f"Mock search result: mechanism framing for {domain.lower()}",
            abstract=(
                f"This mock abstract explores how a context-sensitive mechanism might explain the question "
                f"'{base_title}'. It emphasizes moderator variables, competing mechanisms, and falsifiable predictions."
            ),
            source="Mock Literature Search",
            publication_year=2024,
            doi="",
            openalex_id="mock-openalex-1",
            authors=["Hypothesis Forge Demo"],
            landing_page_url="",
        ),
        LiteratureSearchHit(
            title=f"Mock search result: discriminating experiments in {domain.lower()}",
            abstract=(
                f"This mock abstract proposes experiments that distinguish between broad correlation-based explanations "
                f"and sharper mechanism-level hypotheses for '{base_title}'."
            ),
            source="Mock Literature Search",
            publication_year=2023,
            doi="",
            openalex_id="mock-openalex-2",
            authors=["Hypothesis Forge Demo"],
            landing_page_url="",
        ),
    ][: inputs.max_literature_results]


def _mock_gap_generation(
    inputs: PipelineInputs,
    evidence_items: list[EvidenceItem],
    dataset_summaries: list[DatasetSummary],
    image_observations: list[ImageObservation],
) -> list[KnowledgeGap]:
    sources = preview_list([item.citation for item in evidence_items] + [d.filename for d in dataset_summaries], limit=3)
    domain_focus = inputs.domain_notes or inputs.domain
    return [
        KnowledgeGap(
            gap_title=f"Mechanistic ambiguity in {domain_focus}",
            description="The collected inputs suggest relevant signals, but they do not isolate a clear causal mechanism.",
            why_it_matters="A mechanism-focused gap helps prevent overinterpreting correlations as causes.",
            supporting_evidence=sources,
            missing_information="Perturbation studies or more granular measurements linking inputs to outcomes.",
            suggested_next_step="Design a controlled comparison that varies one suspected driver at a time.",
        ),
        KnowledgeGap(
            gap_title="Limited cross-modal triangulation",
            description="Literature, tabular patterns, and image observations are not yet tightly aligned around one testable explanation.",
            why_it_matters="Triangulated evidence usually improves confidence in which hypotheses deserve resources.",
            supporting_evidence=preview_list([obs.filename for obs in image_observations] + sources, limit=3),
            missing_information="Shared variables or labels that make PDF claims, datasets, and images directly comparable.",
            suggested_next_step="Create a small harmonized table that maps literature claims to dataset columns and image features.",
        ),
    ]


def _mock_hypotheses(
    inputs: PipelineInputs,
    evidence_items: list[EvidenceItem],
    dataset_summaries: list[DatasetSummary],
    knowledge_gaps: list[KnowledgeGap],
) -> list[Hypothesis]:
    supporting = preview_list([item.claim for item in evidence_items], limit=3)
    dataset_support = preview_list([summary.plain_english_summary for summary in dataset_summaries], limit=2)
    hypotheses: list[Hypothesis] = []
    for index in range(inputs.num_hypotheses):
        gap = knowledge_gaps[index % len(knowledge_gaps)] if knowledge_gaps else None
        hypotheses.append(
            Hypothesis(
                id=f"H-{index + 1}",
                hypothesis=(
                    f"If {inputs.domain.lower()} samples are stratified by a controllable condition tied to "
                    f"{gap.gap_title.lower() if gap else 'the main uncertainty'}, then one subgroup will show a "
                    "repeatable shift in the primary outcome."
                ),
                scientific_rationale=(
                    "This idea combines recurring themes from the uploaded evidence with the strongest observable dataset patterns."
                ),
                evidence_for=supporting + dataset_support,
                evidence_against=[
                    "Current inputs are observational and may contain confounding factors.",
                    "The uploaded materials may not span enough conditions to support generalization.",
                ],
                novelty_claim=(
                    "The novelty lies in testing a specific bridge between the documented gap and the observed tabular signals."
                    if inputs.run_novelty_check
                    else "Novelty check was disabled, so this claim is provisional."
                ),
                testable_prediction="A predefined outcome metric will shift measurably after the condition is manipulated.",
                proposed_experiment="Run a controlled pilot with a baseline arm and one targeted intervention arm.",
                required_measurements=["primary response metric", "baseline covariates", "quality control markers"],
                controls=["negative control", "baseline condition", "measurement replication"],
                expected_result="Intervention-aligned samples will diverge from baseline in the predicted direction.",
                falsification_criteria="No reproducible difference appears after controlling for batch and baseline effects.",
                risks_or_limitations="Effect sizes may be small, and the present evidence mix may underrepresent null findings.",
                confidence_score=max(0.35, min(0.72, 0.48 + index * 0.03)),
            )
        )
    return hypotheses


def _mock_critique(hypothesis: Hypothesis, research_question: str) -> HypothesisCritique:
    seed = stable_seed(hypothesis.id + research_question)
    base = 3 + (seed % 2)
    novelty = 2 + ((seed // 7) % 3)
    return HypothesisCritique(
        groundedness=base,
        novelty=novelty,
        testability=4,
        specificity=3 + ((seed // 13) % 2),
        plausibility=3,
        usefulness=4,
        main_weakness="The hypothesis depends on proxy measurements that may not cleanly capture the mechanism.",
        suggested_revision="Narrow the intervention and define one primary endpoint before running the pilot.",
        should_keep=True,
    )


def _mock_experiment_plan(hypothesis: Hypothesis) -> ExperimentPlan:
    return ExperimentPlan(
        hypothesis_id=hypothesis.id,
        objective=f"Test whether the prediction in {hypothesis.id} appears under a controlled perturbation.",
        variables=["intervention condition", "primary outcome metric", "batch or context covariates"],
        controls=hypothesis.controls,
        procedure=[
            "Define baseline and intervention groups with matched starting conditions.",
            "Apply the intervention while tracking environmental and procedural consistency.",
            "Collect the primary endpoint and a small panel of validation measurements.",
            "Compare groups using a preregistered analysis plan.",
        ],
        measurements=hypothesis.required_measurements,
        expected_outcomes=hypothesis.expected_result,
        failure_modes=[
            "The intervention effect is smaller than assay noise.",
            "Unmeasured confounders distort the observed difference.",
        ],
        approximate_feasibility_level="medium",
        ethical_or_safety_notes="Review lab-specific safety practices and domain-specific ethics before execution.",
    )


def _mock_refined_critique(refined: RefinedHypothesis) -> HypothesisCritique:
    seed = stable_seed(refined.hypothesis_id + refined.improved_hypothesis)
    return HypothesisCritique(
        groundedness=4,
        novelty=4 + (seed % 2),
        testability=4,
        specificity=4 + ((seed // 7) % 2),
        plausibility=3 + ((seed // 13) % 2),
        usefulness=4,
        main_weakness="The refined mechanism is sharper, but it still depends on how cleanly the missing variable can be measured.",
        suggested_revision="Predefine the moderator measurement and comparator mechanism before executing the study.",
        should_keep=True,
    )


def _mock_refined_hypothesis(
    hypothesis: Hypothesis,
    dataset_summaries: list[DatasetSummary],
    knowledge_gaps: list[KnowledgeGap],
    image_observations: list[ImageObservation],
) -> RefinedHypothesis:
    gap_hint = knowledge_gaps[0].gap_title if knowledge_gaps else "the main unresolved mechanism"
    dataset_hint = (
        dataset_summaries[0].plain_english_summary
        if dataset_summaries
        else "No dataset summary was available, so the refinement leans on literature structure."
    )
    image_hint = (
        image_observations[0].visible_patterns[0]
        if image_observations and image_observations[0].visible_patterns
        else "No image-derived cue was available."
    )
    return RefinedHypothesis(
        hypothesis_id=hypothesis.id,
        original_hypothesis=hypothesis.hypothesis,
        improved_hypothesis=(
            f"{hypothesis.hypothesis} Specifically, the effect should emerge only when "
            f"{gap_hint.lower()} interacts with a measurable moderator rather than through a generic shift."
        ),
        why_original_was_insufficient=(
            "The original version was testable but still broad enough that multiple mechanisms could explain the same outcome."
        ),
        hidden_assumption=(
            "It assumed the observed response came from one dominant pathway rather than a context-dependent interaction."
        ),
        sharper_mechanism=(
            "The refined version focuses on an interaction-sensitive mechanism that should only appear under a constrained set of conditions."
        ),
        key_interaction_or_missing_variable=(
            "A missing moderator variable may determine whether the effect is mechanistic or just a correlated proxy."
        ),
        revised_prediction=(
            "Only the subgroup with the moderator present should show the predicted shift, while comparator groups should remain near baseline."
        ),
        mechanism_discriminating_experiment=(
            "Run a factorial design that perturbs the main intervention and the suspected moderator independently."
        ),
        what_result_would_distinguish_mechanisms=(
            "An interaction effect would support the refined mechanism; a uniform shift across all groups would favor a broader alternative explanation."
        ),
        why_this_is_more_novel=(
            f"It turns a broad correlation-seeking hypothesis into a mechanism-discriminating test anchored to {gap_hint.lower()}."
        ),
        residual_uncertainty=(
            f"{dataset_hint} {image_hint} The refinement still depends on whether the moderator can be measured reliably."
        ),
    )


def _build_trace_candidates(
    inputs: PipelineInputs,
    evidence_items: list[EvidenceItem],
    dataset_summaries: list[DatasetSummary],
    image_observations: list[ImageObservation],
    literature_hits: list[LiteratureSearchHit],
) -> tuple[list[EvidenceTrace], list[EvidenceTrace]]:
    support_candidates: list[EvidenceTrace] = []
    counter_candidates: list[EvidenceTrace] = []

    for item in evidence_items:
        support_candidates.append(
            EvidenceTrace(
                source_type="pdf",
                source_name=item.citation,
                relation="supports",
                excerpt=item.claim,
                rationale=f"Literature evidence suggests mechanism: {item.mechanism}",
            )
        )
        counter_candidates.append(
            EvidenceTrace(
                source_type="pdf",
                source_name=item.citation,
                relation="against",
                excerpt=item.limitation,
                rationale="The same source also reports a limitation or uncertainty.",
            )
        )

    for summary in dataset_summaries:
        support_candidates.append(
            EvidenceTrace(
                source_type="csv",
                source_name=summary.filename,
                relation="supports",
                excerpt=summary.plain_english_summary,
                rationale="Observed dataset patterns may align with the proposed mechanism.",
            )
        )
        counter_candidates.append(
            EvidenceTrace(
                source_type="csv",
                source_name=summary.filename,
                relation="against",
                excerpt=(
                    f"Missing values: {summary.missing_values}. "
                    f"Outliers: {summary.outlier_counts}."
                ),
                rationale="Missingness and outliers can weaken causal interpretation.",
            )
        )

    for observation in image_observations:
        support_candidates.append(
            EvidenceTrace(
                source_type="image",
                source_name=observation.filename,
                relation="supports",
                excerpt=", ".join(observation.visible_patterns) or observation.image_type,
                rationale="Visible structure in the image may be relevant to the proposed explanation.",
            )
        )
        counter_candidates.append(
            EvidenceTrace(
                source_type="image",
                source_name=observation.filename,
                relation="against",
                excerpt=observation.uncertainty,
                rationale="Image interpretation still carries uncertainty.",
            )
        )

    for hit in literature_hits:
        source_label = _literature_hit_label(hit)
        support_candidates.append(
            EvidenceTrace(
                source_type="literature_search",
                source_name=source_label,
                relation="supports",
                excerpt=hit.abstract[:400],
                rationale="Externally searched literature provides additional supporting context.",
            )
        )
        counter_candidates.append(
            EvidenceTrace(
                source_type="literature_search",
                source_name=source_label,
                relation="context",
                excerpt=f"Source: {hit.source}. DOI: {hit.doi or 'n/a'}",
                rationale="External literature broadens context but still requires close reading.",
            )
        )

    if inputs.notes.strip():
        support_candidates.append(
            EvidenceTrace(
                source_type="note",
                source_name="Project Notes",
                relation="context",
                excerpt=inputs.notes.strip()[:280],
                rationale="User-provided notes add project context for interpreting the evidence.",
            )
        )

    return support_candidates, counter_candidates


def _select_relevant_traces(
    query_text: str,
    candidates: list[EvidenceTrace],
    top_k: int = 3,
) -> list[EvidenceTrace]:
    scored = sorted(
        candidates,
        key=lambda trace: (
            _keyword_overlap_score(query_text, f"{trace.excerpt} {trace.rationale} {trace.source_name}"),
            len(trace.excerpt),
        ),
        reverse=True,
    )
    traces = [trace for trace in scored[:top_k] if trace.excerpt]
    return traces or scored[:top_k]


def _report_sections_mock(
    result_inputs: PipelineInputs,
    evidence_items: list[EvidenceItem],
    dataset_summaries: list[DatasetSummary],
    image_observations: list[ImageObservation],
    knowledge_gaps: list[KnowledgeGap],
    hypotheses: list[Hypothesis],
) -> FinalReportSections:
    evidence_summary = (
        f"{len(evidence_items)} evidence items were extracted from the uploaded PDFs. "
        "Most findings should be treated as preliminary because the workflow is synthesis-first."
    )
    dataset_summary = (
        " | ".join(summary.plain_english_summary for summary in dataset_summaries[:3])
        if dataset_summaries
        else "No CSV datasets were analyzed."
    )
    image_summary = (
        " | ".join(obs.uncertainty if obs.skipped else f"{obs.filename}: {', '.join(obs.visible_patterns)}" for obs in image_observations[:3])
        if image_observations
        else "No images were analyzed."
    )
    return FinalReportSections(
        evidence_summary=evidence_summary,
        dataset_summary=dataset_summary,
        image_summary=image_summary,
        limitations=[
            "Generated hypotheses are research ideas, not verified conclusions.",
            "Uploaded inputs may be incomplete, biased, or mismatched in scope.",
            "Mock mode uses deterministic stand-ins instead of live model reasoning.",
        ],
        failure_modes=[
            "Literature chunks may omit context from full papers.",
            "Simple exploratory statistics can surface spurious patterns.",
            "Image observations may be shallow when the vision model is unavailable.",
        ],
        recommended_next_steps=[
            knowledge_gaps[0].suggested_next_step if knowledge_gaps else "Collect richer mechanistic evidence.",
            "Prioritize one top-ranked hypothesis for a low-cost pilot experiment.",
            f"Re-run the pipeline after refining the question: {result_inputs.research_question}",
        ],
    )


def build_markdown_report(
    inputs: PipelineInputs,
    result: PipelineResult,
    report_sections: FinalReportSections,
) -> str:
    lines = [
        f"# {inputs.project_title}",
        "",
        f"**Research Question:** {inputs.research_question}",
        f"**Domain:** {inputs.domain}",
        f"**Mode:** {'Mock / Demo' if result.mock_mode else 'NVIDIA API'}",
        "",
        "## Inputs Analyzed",
        f"- PDFs: {', '.join(result.inputs_analyzed.get('pdfs', [])) or 'None'}",
        f"- External literature hits: {', '.join(result.inputs_analyzed.get('literature_search', [])) or 'None'}",
        f"- CSVs: {', '.join(result.inputs_analyzed.get('csvs', [])) or 'None'}",
        f"- Images: {', '.join(result.inputs_analyzed.get('images', [])) or 'None'}",
        "",
        "## External Literature Search",
    ]
    if result.literature_hits:
        for hit in result.literature_hits[:10]:
            lines.extend(
                [
                    f"### {_literature_hit_label(hit)}",
                    f"- Source: {hit.source}",
                    f"- DOI: {hit.doi or 'n/a'}",
                    f"- Authors: {', '.join(hit.authors[:5]) or 'n/a'}",
                    f"- Abstract excerpt: {hit.abstract[:280]}",
                ]
            )
    else:
        lines.append("No external literature hits were analyzed.")
    lines.extend(
        [
            "",
        "## Evidence Summary",
        report_sections.evidence_summary,
        "",
        "## Dataset Summary",
        report_sections.dataset_summary,
        "",
        "## Image Summary",
        report_sections.image_summary,
        "",
        "## Top Knowledge Gaps",
        ]
    )
    for gap in result.knowledge_gaps[:5]:
        lines.extend(
            [
                f"### {gap.gap_title}",
                gap.description,
                f"- Why it matters: {gap.why_it_matters}",
                f"- Missing information: {gap.missing_information}",
                f"- Suggested next step: {gap.suggested_next_step}",
            ]
        )
    lines.append("")
    lines.append("## Ranked Hypotheses")
    for hypothesis in result.hypotheses:
        lines.extend(
            [
                f"### {hypothesis.id}: {hypothesis.hypothesis}",
                f"- Weighted score: {hypothesis.weighted_score if hypothesis.weighted_score is not None else 'n/a'}",
                f"- Rationale: {hypothesis.scientific_rationale}",
                f"- Prediction: {hypothesis.testable_prediction}",
                f"- Falsification: {hypothesis.falsification_criteria}",
                f"- Risks: {hypothesis.risks_or_limitations}",
                f"- Supporting sources: {', '.join(sorted({trace.source_name for trace in hypothesis.support_traces})) or 'n/a'}",
                f"- Counter-evidence sources: {', '.join(sorted({trace.source_name for trace in hypothesis.counter_traces})) or 'n/a'}",
            ]
        )
    lines.append("")
    lines.append("## Refined Hypotheses")
    if result.refined_hypotheses:
        for refined in result.refined_hypotheses:
            lines.extend(
                [
                    f"### {refined.hypothesis_id}",
                    f"- Original hypothesis: {refined.original_hypothesis}",
                    f"- Improved hypothesis: {refined.improved_hypothesis}",
                    f"- Refined weighted score: {refined.weighted_score if refined.weighted_score is not None else 'n/a'}",
                    f"- Score delta vs original: {refined.score_delta_vs_original if refined.score_delta_vs_original is not None else 'n/a'}",
                    f"- Why the original was insufficient: {refined.why_original_was_insufficient}",
                    f"- Sharper mechanism: {refined.sharper_mechanism}",
                    f"- Revised prediction: {refined.revised_prediction}",
                    f"- Mechanism-discriminating experiment: {refined.mechanism_discriminating_experiment}",
                    f"- Distinguishing result: {refined.what_result_would_distinguish_mechanisms}",
                    f"- Novelty gain: {refined.why_this_is_more_novel}",
                    f"- Residual uncertainty: {refined.residual_uncertainty}",
                    f"- Supporting sources: {', '.join(sorted({trace.source_name for trace in refined.support_traces})) or 'n/a'}",
                    f"- Counter-evidence sources: {', '.join(sorted({trace.source_name for trace in refined.counter_traces})) or 'n/a'}",
                ]
            )
    else:
        lines.append("No refined hypotheses were generated.")
    lines.append("")
    lines.append("## Experiment Plans")
    for plan in result.experiment_plans:
        lines.extend(
            [
                f"### {plan.hypothesis_id}",
                f"- Objective: {plan.objective}",
                f"- Variables: {', '.join(plan.variables)}",
                f"- Controls: {', '.join(plan.controls)}",
                f"- Measurements: {', '.join(plan.measurements)}",
                f"- Feasibility: {plan.approximate_feasibility_level}",
                f"- Safety / ethics: {plan.ethical_or_safety_notes}",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            *[f"- {item}" for item in report_sections.limitations],
            "",
            "## Failure Modes",
            *[f"- {item}" for item in report_sections.failure_modes],
            "",
            "## Recommended Next Steps",
            *[f"- {item}" for item in report_sections.recommended_next_steps],
        ]
    )
    return "\n".join(lines)


def _validate_model_output(model_cls: Any, payload: Any, key: str | None = None) -> Any:
    candidate = payload[key] if key else payload
    if isinstance(candidate, list):
        return [model_cls.model_validate(item) for item in candidate]
    return model_cls.model_validate(candidate)


class CoScientistPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = NvidiaClient(self.settings)
        self.literature_searcher = OpenAlexLiteratureSearcher()

    @staticmethod
    def _notify_progress(
        progress_callback: ProgressCallback | None,
        value: int,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    def _embed_query(self, text: str) -> list[float]:
        if self.settings.embed_enabled:
            return self.client.embed_texts([text])[0]
        return _mock_embedding(text)

    def _chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raw = self.client.chat_completion(messages, response_format_json=True)
        return parse_json_with_repair(raw)

    def _search_literature(
        self,
        inputs: PipelineInputs,
        warnings: list[str],
    ) -> list[LiteratureSearchHit]:
        if not inputs.run_literature_search:
            return []
        try:
            hits = self.literature_searcher.search(
                query=inputs.research_question,
                max_results=inputs.max_literature_results,
                domain=inputs.domain,
                domain_notes=inputs.domain_notes,
            )
            if not hits:
                if self.settings.is_mock_mode:
                    warnings.append(
                        "External literature search returned no abstract-bearing results, so demo fallback papers were used."
                    )
                    return _mock_literature_hits(inputs)
                warnings.append(
                    "External literature search ran successfully but returned no abstract-bearing results for this query."
                )
            return hits
        except Exception as exc:
            if self.settings.is_mock_mode:
                warnings.append(
                    "External literature search could not be reached, so demo fallback papers were used."
                )
                warnings.append(f"Search error details: {exc}")
                return _mock_literature_hits(inputs)
            warnings.append(f"External literature search was skipped: {exc}")
            return []

    def _extract_evidence(
        self,
        pdfs: list[UploadedArtifact],
        literature_hits: list[LiteratureSearchHit],
        research_question: str,
        warnings: list[str],
    ) -> tuple[list[EvidenceItem], LocalRetriever]:
        retriever = LocalRetriever(query_embedder=self._embed_query)
        evidence_items: list[EvidenceItem] = []
        retrieval_docs: list[RetrievalDocument] = []
        embeddings: list[list[float]] = []

        for pdf in pdfs:
            try:
                chunks = extract_chunks_from_pdf(pdf.data)
            except Exception as exc:  # pragma: no cover - depends on uploaded content
                warnings.append(f"Could not read PDF {pdf.name}: {exc}")
                continue
            for index, chunk in enumerate(chunks):
                retrieval_docs.append(
                    RetrievalDocument(
                        text=chunk,
                        source=pdf.name,
                        metadata={"chunk_index": index},
                    )
                )
                embeddings.append(_mock_embedding(chunk))
                if self.settings.is_mock_mode:
                    evidence_items.extend(_mock_evidence_from_chunk(chunk, pdf.name, index))
                    continue
                try:
                    payload = self._chat_json(
                        agents.literature_evidence_prompt(chunk, pdf.name, research_question)
                    )
                    evidence_items.extend(_validate_model_output(EvidenceItem, payload, "evidence_items"))
                except (ValidationError, KeyError, Exception) as exc:
                    warnings.append(f"Evidence extraction fallback used for {pdf.name} chunk {index + 1}: {exc}")
                    evidence_items.extend(_mock_evidence_from_chunk(chunk, pdf.name, index))

        for index, hit in enumerate(literature_hits):
            chunk = hit.abstract.strip()
            if not chunk:
                continue
            source_label = _literature_hit_label(hit)
            retrieval_docs.append(
                RetrievalDocument(
                    text=chunk,
                    source=source_label,
                    metadata={
                        "source_type": "literature_search",
                        "openalex_id": hit.openalex_id,
                        "doi": hit.doi,
                    },
                )
            )
            embeddings.append(_mock_embedding(chunk))
            if self.settings.is_mock_mode:
                evidence_items.extend(_mock_evidence_from_chunk(chunk, source_label, index))
                continue
            try:
                payload = self._chat_json(
                    agents.literature_evidence_prompt(chunk, source_label, research_question)
                )
                evidence_items.extend(_validate_model_output(EvidenceItem, payload, "evidence_items"))
            except (ValidationError, KeyError, Exception) as exc:
                warnings.append(f"Search-hit evidence fallback used for {source_label}: {exc}")
                evidence_items.extend(_mock_evidence_from_chunk(chunk, source_label, index))

        if retrieval_docs:
            live_embeddings: list[list[float]] | None = None
            if self.settings.embed_enabled:
                try:
                    live_embeddings = self.client.embed_texts([doc.text for doc in retrieval_docs])
                except Exception as exc:  # pragma: no cover - external API
                    warnings.append(
                        f"Embedding endpoint unavailable; using keyword retrieval instead: {exc}"
                    )
            retriever.add_documents(
                retrieval_docs,
                embeddings if self.settings.is_mock_mode else live_embeddings,
            )
        return evidence_items, retriever

    def _process_datasets(self, csvs: list[UploadedArtifact], research_question: str, warnings: list[str]) -> list[DatasetSummary]:
        summaries: list[DatasetSummary] = []
        for csv_file in csvs:
            try:
                summary = analyze_csv_bytes(csv_file.data, csv_file.name)
            except Exception as exc:
                warnings.append(f"Could not parse CSV {csv_file.name}: {exc}")
                continue
            if not self.settings.is_mock_mode:
                try:
                    payload = self._chat_json(
                        agents.data_interpretation_prompt(summary.model_dump(), research_question)
                    )
                    summary.plain_english_summary = str(payload.get("plain_english_summary", summary.plain_english_summary))
                except Exception as exc:
                    warnings.append(f"Dataset interpretation fallback used for {csv_file.name}: {exc}")
            summaries.append(summary)
        return summaries

    def _process_images(
        self,
        images: list[UploadedArtifact],
        research_question: str,
        include_image_analysis: bool,
        warnings: list[str],
    ) -> list[ImageObservation]:
        observations: list[ImageObservation] = []
        for image in images:
            if not include_image_analysis:
                observations.append(fallback_image_observation(image.data, image.name, skipped=True))
                continue
            if not self.settings.vision_enabled:
                observations.append(fallback_image_observation(image.data, image.name, skipped=True))
                continue
            try:
                payload = parse_json_with_repair(
                    self.client.analyze_image(
                        image.data,
                        agents.image_interpretation_prompt(image.name, research_question),
                    )
                )
                observations.append(ImageObservation.model_validate(payload))
            except Exception as exc:
                warnings.append(f"Image analysis fallback used for {image.name}: {exc}")
                observations.append(fallback_image_observation(image.data, image.name, skipped=True))
        return observations

    def _generate_gaps(
        self,
        inputs: PipelineInputs,
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        image_observations: list[ImageObservation],
        literature_hits: list[LiteratureSearchHit],
        warnings: list[str],
    ) -> list[KnowledgeGap]:
        if self.settings.is_mock_mode:
            return _mock_gap_generation(inputs, evidence_items, dataset_summaries, image_observations)
        context = {
            "research_question": inputs.research_question,
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
            "literature_hits": [item.model_dump() for item in literature_hits[:6]],
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
            "notes": inputs.notes,
        }
        try:
            payload = self._chat_json(agents.gap_detection_prompt(context))
            return _validate_model_output(KnowledgeGap, payload, "knowledge_gaps")
        except Exception as exc:
            warnings.append(f"Gap detection fallback used: {exc}")
            return _mock_gap_generation(inputs, evidence_items, dataset_summaries, image_observations)

    def _generate_hypotheses(
        self,
        inputs: PipelineInputs,
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        image_observations: list[ImageObservation],
        literature_hits: list[LiteratureSearchHit],
        knowledge_gaps: list[KnowledgeGap],
        retriever: LocalRetriever,
        warnings: list[str],
    ) -> list[Hypothesis]:
        if self.settings.is_mock_mode:
            return _mock_hypotheses(inputs, evidence_items, dataset_summaries, knowledge_gaps)
        context = {
            "research_question": inputs.research_question,
            "knowledge_gaps": [gap.model_dump() for gap in knowledge_gaps],
            "supporting_evidence": [item.model_dump() for item in evidence_items[:10]],
            "literature_hits": [item.model_dump() for item in literature_hits[:6]],
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
            "retrieved_chunks": [doc.model_dump() for doc in retriever.query(inputs.research_question, top_k=5)],
            "notes": inputs.notes,
        }
        try:
            payload = self._chat_json(agents.hypothesis_generation_prompt(context, inputs.num_hypotheses))
            return _validate_model_output(Hypothesis, payload, "hypotheses")
        except Exception as exc:
            warnings.append(f"Hypothesis generation fallback used: {exc}")
            return _mock_hypotheses(inputs, evidence_items, dataset_summaries, knowledge_gaps)

    def _critique_hypotheses(
        self,
        inputs: PipelineInputs,
        hypotheses: list[Hypothesis],
        evidence_items: list[EvidenceItem],
        warnings: list[str],
    ) -> list[Hypothesis]:
        context = {
            "research_question": inputs.research_question,
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
        }
        for hypothesis in hypotheses:
            if not inputs.run_skeptical_critic:
                hypothesis.critique = HypothesisCritique(
                    groundedness=3,
                    novelty=3,
                    testability=3,
                    specificity=3,
                    plausibility=3,
                    usefulness=3,
                    main_weakness="Skeptical critic disabled.",
                    suggested_revision="Enable the critic to get structured feedback.",
                    should_keep=True,
                )
                continue
            if self.settings.is_mock_mode:
                hypothesis.critique = _mock_critique(hypothesis, inputs.research_question)
                continue
            try:
                payload = self._chat_json(
                    agents.hypothesis_critique_prompt(hypothesis.model_dump(), context)
                )
                hypothesis.critique = HypothesisCritique.model_validate(payload)
            except Exception as exc:
                warnings.append(f"Critique fallback used for {hypothesis.id}: {exc}")
                hypothesis.critique = _mock_critique(hypothesis, inputs.research_question)
        return rank_hypotheses(hypotheses)

    def _attach_traces_to_hypotheses(
        self,
        inputs: PipelineInputs,
        hypotheses: list[Hypothesis],
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        image_observations: list[ImageObservation],
        literature_hits: list[LiteratureSearchHit],
    ) -> list[Hypothesis]:
        support_candidates, counter_candidates = _build_trace_candidates(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            literature_hits,
        )
        for hypothesis in hypotheses:
            query_text = " ".join(
                [
                    hypothesis.hypothesis,
                    hypothesis.scientific_rationale,
                    " ".join(hypothesis.evidence_for),
                    " ".join(hypothesis.evidence_against),
                ]
            )
            hypothesis.support_traces = _select_relevant_traces(query_text, support_candidates, top_k=3)
            hypothesis.counter_traces = _select_relevant_traces(query_text, counter_candidates, top_k=2)
        return hypotheses

    def _design_experiments(
        self,
        hypotheses: list[Hypothesis],
        evidence_items: list[EvidenceItem],
        warnings: list[str],
    ) -> list[ExperimentPlan]:
        top_hypotheses = hypotheses[:]
        context = {"evidence_items": [item.model_dump() for item in evidence_items[:6]]}
        plans: list[ExperimentPlan] = []
        for hypothesis in top_hypotheses:
            if self.settings.is_mock_mode:
                plans.append(_mock_experiment_plan(hypothesis))
                continue
            try:
                payload = self._chat_json(
                    agents.experiment_design_prompt(hypothesis.model_dump(), context)
                )
                payload["hypothesis_id"] = hypothesis.id
                plans.append(ExperimentPlan.model_validate(payload))
            except Exception as exc:
                warnings.append(f"Experiment plan fallback used for {hypothesis.id}: {exc}")
                plans.append(_mock_experiment_plan(hypothesis))
        return plans

    def _refine_hypotheses(
        self,
        inputs: PipelineInputs,
        hypotheses: list[Hypothesis],
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        knowledge_gaps: list[KnowledgeGap],
        image_observations: list[ImageObservation],
        warnings: list[str],
    ) -> list[RefinedHypothesis]:
        if not inputs.run_hypothesis_refinement:
            return []
        refined: list[RefinedHypothesis] = []
        context = {
            "research_question": inputs.research_question,
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "knowledge_gaps": [item.model_dump() for item in knowledge_gaps[:5]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
        }
        for hypothesis in hypotheses:
            if self.settings.is_mock_mode:
                mock_refined = _mock_refined_hypothesis(
                    hypothesis,
                    dataset_summaries,
                    knowledge_gaps,
                    image_observations,
                )
                mock_refined.support_traces = list(hypothesis.support_traces)
                mock_refined.counter_traces = list(hypothesis.counter_traces)
                refined.append(mock_refined)
                continue
            hypothesis_payload = hypothesis.model_dump()
            try:
                payload = self._chat_json(
                    agents.hypothesis_refinement_prompt(hypothesis_payload, context)
                )
                payload["hypothesis_id"] = hypothesis.id
                payload["original_hypothesis"] = hypothesis.hypothesis
                refined_item = RefinedHypothesis.model_validate(payload)
                refined_item.support_traces = list(hypothesis.support_traces)
                refined_item.counter_traces = list(hypothesis.counter_traces)
                refined.append(refined_item)
            except Exception as exc:
                warnings.append(f"Refinement fallback used for {hypothesis.id}: {exc}")
                mock_refined = _mock_refined_hypothesis(
                    hypothesis,
                    dataset_summaries,
                    knowledge_gaps,
                    image_observations,
                )
                mock_refined.support_traces = list(hypothesis.support_traces)
                mock_refined.counter_traces = list(hypothesis.counter_traces)
                refined.append(mock_refined)
        return refined

    def _recritique_refined_hypotheses(
        self,
        inputs: PipelineInputs,
        refined_hypotheses: list[RefinedHypothesis],
        original_hypotheses: list[Hypothesis],
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        knowledge_gaps: list[KnowledgeGap],
        image_observations: list[ImageObservation],
        warnings: list[str],
    ) -> list[RefinedHypothesis]:
        if not refined_hypotheses:
            return []
        context = {
            "research_question": inputs.research_question,
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "knowledge_gaps": [item.model_dump() for item in knowledge_gaps[:5]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
        }
        for refined in refined_hypotheses:
            if self.settings.is_mock_mode:
                refined.refined_critique = _mock_refined_critique(refined)
                continue
            try:
                payload = self._chat_json(
                    agents.refined_hypothesis_critique_prompt(refined.model_dump(), context)
                )
                refined.refined_critique = HypothesisCritique.model_validate(payload)
            except Exception as exc:
                warnings.append(f"Refined critique fallback used for {refined.hypothesis_id}: {exc}")
                refined.refined_critique = _mock_refined_critique(refined)
        return score_refined_hypotheses(refined_hypotheses, original_hypotheses)

    def _final_sections(
        self,
        inputs: PipelineInputs,
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        image_observations: list[ImageObservation],
        literature_hits: list[LiteratureSearchHit],
        knowledge_gaps: list[KnowledgeGap],
        hypotheses: list[Hypothesis],
        refined_hypotheses: list[RefinedHypothesis],
        warnings: list[str],
    ) -> FinalReportSections:
        if self.settings.is_mock_mode:
            return _report_sections_mock(
                inputs,
                evidence_items,
                dataset_summaries,
                image_observations,
                knowledge_gaps,
                hypotheses,
            )
        context = {
            "project_title": inputs.project_title,
            "research_question": inputs.research_question,
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
            "literature_hits": [item.model_dump() for item in literature_hits[:6]],
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
            "knowledge_gaps": [item.model_dump() for item in knowledge_gaps[:5]],
            "hypotheses": [item.model_dump() for item in hypotheses[:5]],
            "refined_hypotheses": [item.model_dump() for item in refined_hypotheses[:5]],
        }
        try:
            payload = self._chat_json(agents.final_report_prompt(context))
            return FinalReportSections.model_validate(payload)
        except Exception as exc:
            warnings.append(f"Final report fallback used: {exc}")
            return _report_sections_mock(
                inputs,
                evidence_items,
                dataset_summaries,
                image_observations,
                knowledge_gaps,
                hypotheses,
            )

    def run(
        self,
        inputs: PipelineInputs,
        pdfs: list[UploadedArtifact],
        csvs: list[UploadedArtifact],
        images: list[UploadedArtifact],
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        warnings: list[str] = []
        self._notify_progress(progress_callback, 5, "Searching external literature...")
        literature_hits = self._search_literature(inputs, warnings)
        self._notify_progress(progress_callback, 18, "Reading PDFs and extracting evidence...")
        evidence_items, retriever = self._extract_evidence(
            pdfs,
            literature_hits,
            inputs.research_question,
            warnings,
        )
        self._notify_progress(progress_callback, 33, "Profiling uploaded CSV datasets...")
        dataset_summaries = self._process_datasets(csvs, inputs.research_question, warnings)
        self._notify_progress(progress_callback, 48, "Reviewing uploaded images...")
        image_observations = self._process_images(
            images,
            inputs.research_question,
            inputs.include_image_analysis,
            warnings,
        )
        self._notify_progress(progress_callback, 62, "Identifying knowledge gaps...")
        knowledge_gaps = self._generate_gaps(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            literature_hits,
            warnings,
        )
        self._notify_progress(progress_callback, 76, "Generating candidate hypotheses...")
        hypotheses = self._generate_hypotheses(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            literature_hits,
            knowledge_gaps,
            retriever,
            warnings,
        )
        self._notify_progress(progress_callback, 85, "Critiquing and ranking hypotheses...")
        hypotheses = self._critique_hypotheses(inputs, hypotheses, evidence_items, warnings)
        top_hypotheses = hypotheses[: inputs.top_k]
        top_hypotheses = self._attach_traces_to_hypotheses(
            inputs,
            top_hypotheses,
            evidence_items,
            dataset_summaries,
            image_observations,
            literature_hits,
        )
        self._notify_progress(progress_callback, 90, "Refining top hypotheses...")
        refined_hypotheses = self._refine_hypotheses(
            inputs,
            top_hypotheses,
            evidence_items,
            dataset_summaries,
            knowledge_gaps,
            image_observations,
            warnings,
        )
        self._notify_progress(progress_callback, 93, "Re-critique refined hypotheses...")
        refined_hypotheses = self._recritique_refined_hypotheses(
            inputs,
            refined_hypotheses,
            top_hypotheses,
            evidence_items,
            dataset_summaries,
            knowledge_gaps,
            image_observations,
            warnings,
        )
        self._notify_progress(progress_callback, 96, "Designing experiment plans...")
        experiment_plans = self._design_experiments(top_hypotheses, evidence_items, warnings)
        self._notify_progress(progress_callback, 99, "Writing the final report...")
        report_sections = self._final_sections(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            literature_hits,
            knowledge_gaps,
            top_hypotheses,
            refined_hypotheses,
            warnings,
        )
        result = PipelineResult(
            project_title=inputs.project_title,
            research_question=inputs.research_question,
            domain=inputs.domain,
            mock_mode=self.settings.is_mock_mode,
            inputs_analyzed={
                "pdfs": [pdf.name for pdf in pdfs],
                "csvs": [csv.name for csv in csvs],
                "images": [image.name for image in images],
                "literature_search": [_literature_hit_label(hit) for hit in literature_hits],
            },
            literature_hits=literature_hits,
            evidence_items=evidence_items,
            dataset_summaries=dataset_summaries,
            image_observations=image_observations,
            knowledge_gaps=knowledge_gaps,
            hypotheses=top_hypotheses,
            refined_hypotheses=refined_hypotheses,
            experiment_plans=experiment_plans,
            final_report_markdown="",
            warnings=warnings,
        )
        result.final_report_markdown = build_markdown_report(inputs, result, report_sections)
        self._notify_progress(progress_callback, 100, "Pipeline complete.")
        return result
