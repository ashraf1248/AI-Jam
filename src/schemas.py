from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip() or "H-0"


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
    run_novelty_check: bool = True
    run_skeptical_critic: bool = True


class PipelineResult(BaseModel):
    project_title: str
    research_question: str
    domain: str
    mock_mode: bool
    inputs_analyzed: dict[str, list[str]]
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    dataset_summaries: list[DatasetSummary] = Field(default_factory=list)
    image_observations: list[ImageObservation] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
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
