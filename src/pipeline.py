from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src import agents
from src.config import Settings
from src.data_processor import analyze_csv_bytes
from src.export import export_markdown_report
from src.image_processor import fallback_image_observation
from src.nvidia_client import NvidiaClient
from src.pdf_processor import extract_chunks_from_pdf
from src.retrieval import LocalRetriever, _mock_embedding
from src.schemas import (
    DatasetSummary,
    EvidenceItem,
    ExperimentPlan,
    FinalReportSections,
    Hypothesis,
    HypothesisCritique,
    ImageObservation,
    KnowledgeGap,
    PipelineInputs,
    PipelineResult,
    RetrievalDocument,
)
from src.scoring import rank_hypotheses
from src.utils import parse_json_with_repair, preview_list, stable_seed


@dataclass(slots=True)
class UploadedArtifact:
    name: str
    data: bytes


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
        f"- CSVs: {', '.join(result.inputs_analyzed.get('csvs', [])) or 'None'}",
        f"- Images: {', '.join(result.inputs_analyzed.get('images', [])) or 'None'}",
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
            ]
        )
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

    def _chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raw = self.client.chat_completion(messages, response_format_json=True)
        return parse_json_with_repair(raw)

    def _extract_evidence(self, pdfs: list[UploadedArtifact], research_question: str, warnings: list[str]) -> tuple[list[EvidenceItem], LocalRetriever]:
        retriever = LocalRetriever()
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

        if retrieval_docs:
            live_embeddings: list[list[float]] | None = None
            if self.settings.embed_enabled:
                try:
                    live_embeddings = self.client.embed_texts([doc.text for doc in retrieval_docs])
                except Exception as exc:  # pragma: no cover - external API
                    warnings.append(f"Embedding request failed; using keyword retrieval instead: {exc}")
            retriever.add_documents(retrieval_docs, live_embeddings or embeddings)
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
        warnings: list[str],
    ) -> list[KnowledgeGap]:
        if self.settings.is_mock_mode:
            return _mock_gap_generation(inputs, evidence_items, dataset_summaries, image_observations)
        context = {
            "research_question": inputs.research_question,
            "evidence_items": [item.model_dump() for item in evidence_items[:8]],
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

    def _final_sections(
        self,
        inputs: PipelineInputs,
        evidence_items: list[EvidenceItem],
        dataset_summaries: list[DatasetSummary],
        image_observations: list[ImageObservation],
        knowledge_gaps: list[KnowledgeGap],
        hypotheses: list[Hypothesis],
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
            "dataset_summaries": [item.model_dump() for item in dataset_summaries[:4]],
            "image_observations": [item.model_dump() for item in image_observations[:4]],
            "knowledge_gaps": [item.model_dump() for item in knowledge_gaps[:5]],
            "hypotheses": [item.model_dump() for item in hypotheses[:5]],
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
    ) -> PipelineResult:
        warnings: list[str] = []
        evidence_items, retriever = self._extract_evidence(pdfs, inputs.research_question, warnings)
        dataset_summaries = self._process_datasets(csvs, inputs.research_question, warnings)
        image_observations = self._process_images(
            images,
            inputs.research_question,
            inputs.include_image_analysis,
            warnings,
        )
        knowledge_gaps = self._generate_gaps(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            warnings,
        )
        hypotheses = self._generate_hypotheses(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            knowledge_gaps,
            retriever,
            warnings,
        )
        hypotheses = self._critique_hypotheses(inputs, hypotheses, evidence_items, warnings)
        top_hypotheses = hypotheses[: inputs.top_k]
        experiment_plans = self._design_experiments(top_hypotheses, evidence_items, warnings)
        report_sections = self._final_sections(
            inputs,
            evidence_items,
            dataset_summaries,
            image_observations,
            knowledge_gaps,
            top_hypotheses,
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
            },
            evidence_items=evidence_items,
            dataset_summaries=dataset_summaries,
            image_observations=image_observations,
            knowledge_gaps=knowledge_gaps,
            hypotheses=top_hypotheses,
            experiment_plans=experiment_plans,
            final_report_markdown="",
            warnings=warnings,
        )
        result.final_report_markdown = build_markdown_report(inputs, result, report_sections)
        return result
