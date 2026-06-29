from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _flatten_string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        items: list[str] = []
        for key, nested_value in value.items():
            nested_items = _flatten_string_items(nested_value)
            label = str(key).strip()
            if nested_items:
                items.append(f"{label}: {', '.join(nested_items)}")
            elif label:
                items.append(label)
        return items
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_string_items(item))
        return items
    return [str(value)]


class EvidenceItem(BaseModel):
    claim: str
    mechanism: str
    variables: list[str] = Field(default_factory=list)
    method: str
    limitation: str
    citation: str
    confidence: float = Field(ge=0.0, le=1.0)


class DatasetCorrelation(BaseModel):
    column_a: str
    column_b: str
    correlation: float


class RegressionInsight(BaseModel):
    target_column: str
    feature_importance: dict[str, float] = Field(default_factory=dict)
    r2_score: float | None = None


class DatasetSummary(BaseModel):
    filename: str
    shape: tuple[int, int]
    columns: list[str]
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    missing_values: dict[str, int] = Field(default_factory=dict)
    descriptive_statistics: dict[str, dict[str, float | int | None]] = Field(
        default_factory=dict
    )
    strongest_correlations: list[DatasetCorrelation] = Field(default_factory=list)
    outlier_counts: dict[str, int] = Field(default_factory=dict)
    regression_insight: RegressionInsight | None = None
    plain_english_summary: str


class ImageObservation(BaseModel):
    filename: str
    image_type: str
    visible_patterns: list[str] = Field(default_factory=list)
    possible_measurements: list[str] = Field(default_factory=list)
    uncertainty: str
    skipped: bool = False


class KnowledgeGap(BaseModel):
    gap_title: str
    description: str
    why_it_matters: str
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_information: str
    suggested_next_step: str


class HypothesisCritique(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    testability: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    plausibility: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    main_weakness: str
    suggested_revision: str
    should_keep: bool = True


class EvidenceTrace(BaseModel):
    source_type: Literal["pdf", "csv", "image", "note", "literature_search"]
    source_name: str
    relation: Literal["supports", "against", "context"]
    excerpt: str
    rationale: str


class LiteratureSearchHit(BaseModel):
    title: str
    abstract: str
    source: str
    publication_year: int | None = None
    doi: str = ""
    openalex_id: str = ""
    authors: list[str] = Field(default_factory=list)
    landing_page_url: str = ""


class Hypothesis(BaseModel):
    id: str
    hypothesis: str
    scientific_rationale: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    novelty_claim: str
    testable_prediction: str
    proposed_experiment: str
    required_measurements: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    expected_result: str
    falsification_criteria: str
    risks_or_limitations: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    critique: HypothesisCritique | None = None
    weighted_score: float | None = None
    support_traces: list[EvidenceTrace] = Field(default_factory=list)
    counter_traces: list[EvidenceTrace] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip() or "H-0"


class RefinedHypothesis(BaseModel):
    hypothesis_id: str
    original_hypothesis: str
    improved_hypothesis: str
    why_original_was_insufficient: str
    hidden_assumption: str
    sharper_mechanism: str
    key_interaction_or_missing_variable: str
    revised_prediction: str
    mechanism_discriminating_experiment: str
    what_result_would_distinguish_mechanisms: str
    why_this_is_more_novel: str
    residual_uncertainty: str
    refined_critique: HypothesisCritique | None = None
    weighted_score: float | None = None
    score_delta_vs_original: float | None = None
    support_traces: list[EvidenceTrace] = Field(default_factory=list)
    counter_traces: list[EvidenceTrace] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    hypothesis_id: str
    objective: str
    variables: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    measurements: list[str] = Field(default_factory=list)
    expected_outcomes: str
    failure_modes: list[str] = Field(default_factory=list)
    approximate_feasibility_level: Literal["low", "medium", "high"]
    ethical_or_safety_notes: str

    @field_validator("variables", "controls", "procedure", "measurements", "failure_modes", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return _flatten_string_items(value)


class RetrievalDocument(BaseModel):
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineInputs(BaseModel):
    project_title: str
    research_question: str
    domain: str
    domain_notes: str = ""
    notes: str = ""
    num_hypotheses: int = Field(default=5, ge=1, le=12)
    top_k: int = Field(default=3, ge=1, le=12)
    include_image_analysis: bool = True
    run_literature_search: bool = False
    max_literature_results: int = Field(default=5, ge=1, le=20)
    run_novelty_check: bool = True
    run_skeptical_critic: bool = True
    run_hypothesis_refinement: bool = True


class PipelineResult(BaseModel):
    project_title: str
    research_question: str
    domain: str
    mock_mode: bool
    inputs_analyzed: dict[str, list[str]]
    literature_hits: list[LiteratureSearchHit] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    dataset_summaries: list[DatasetSummary] = Field(default_factory=list)
    image_observations: list[ImageObservation] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    refined_hypotheses: list[RefinedHypothesis] = Field(default_factory=list)
    experiment_plans: list[ExperimentPlan] = Field(default_factory=list)
    final_report_markdown: str
    warnings: list[str] = Field(default_factory=list)


class FinalReportSections(BaseModel):
    evidence_summary: str
    dataset_summary: str
    image_summary: str
    limitations: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class SavedUploadArtifact(BaseModel):
    name: str
    data_base64: str


class SavedProject(BaseModel):
    project_id: str
    project_title: str
    created_at: str
    updated_at: str
    inputs: PipelineInputs
    result: PipelineResult | None = None
    stored_artifacts: dict[str, list[SavedUploadArtifact]] = Field(default_factory=dict)
    upload_manifest: dict[str, list[str]] = Field(default_factory=dict)


class SavedProjectSummary(BaseModel):
    project_id: str
    project_title: str
    updated_at: str
    has_result: bool = False
    upload_manifest: dict[str, list[str]] = Field(default_factory=dict)
