from __future__ import annotations

from src.schemas import Hypothesis, HypothesisCritique, RefinedHypothesis


WEIGHTS = {
    "groundedness": 0.25,
    "testability": 0.25,
    "novelty": 0.20,
    "specificity": 0.15,
    "plausibility": 0.10,
    "usefulness": 0.05,
}


def normalize_scores_from_critique(critique: HypothesisCritique | None) -> dict[str, float]:
    if critique is None:
        return {key: 0.0 for key in WEIGHTS}
    return {
        key: (getattr(critique, key) - 1) / 4
        for key in WEIGHTS
    }


def normalize_critic_scores(hypothesis: Hypothesis) -> dict[str, float]:
    return normalize_scores_from_critique(hypothesis.critique)


def compute_weighted_total_score(hypothesis: Hypothesis) -> float:
    normalized = normalize_critic_scores(hypothesis)
    return sum(normalized[key] * weight for key, weight in WEIGHTS.items())


def compute_weighted_score_from_critique(critique: HypothesisCritique | None) -> float:
    normalized = normalize_scores_from_critique(critique)
    return sum(normalized[key] * weight for key, weight in WEIGHTS.items())


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    ranked: list[Hypothesis] = []
    for hypothesis in hypotheses:
        hypothesis.weighted_score = compute_weighted_total_score(hypothesis)
        ranked.append(hypothesis)
    return sorted(ranked, key=lambda item: item.weighted_score or 0.0, reverse=True)


def score_refined_hypotheses(
    refined_hypotheses: list[RefinedHypothesis],
    original_hypotheses: list[Hypothesis],
) -> list[RefinedHypothesis]:
    original_scores = {
        hypothesis.id: (hypothesis.weighted_score or compute_weighted_total_score(hypothesis))
        for hypothesis in original_hypotheses
    }
    scored: list[RefinedHypothesis] = []
    for refined in refined_hypotheses:
        refined.weighted_score = compute_weighted_score_from_critique(refined.refined_critique)
        original_score = original_scores.get(refined.hypothesis_id, 0.0)
        refined.score_delta_vs_original = refined.weighted_score - original_score
        scored.append(refined)
    return sorted(scored, key=lambda item: item.weighted_score or 0.0, reverse=True)
