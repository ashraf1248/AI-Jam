from __future__ import annotations

from src.schemas import Hypothesis


WEIGHTS = {
    "groundedness": 0.25,
    "testability": 0.25,
    "novelty": 0.20,
    "specificity": 0.15,
    "plausibility": 0.10,
    "usefulness": 0.05,
}


def normalize_critic_scores(hypothesis: Hypothesis) -> dict[str, float]:
    critique = hypothesis.critique
    if critique is None:
        return {key: 0.0 for key in WEIGHTS}
    return {
        key: (getattr(critique, key) - 1) / 4
        for key in WEIGHTS
    }


def compute_weighted_total_score(hypothesis: Hypothesis) -> float:
    normalized = normalize_critic_scores(hypothesis)
    return sum(normalized[key] * weight for key, weight in WEIGHTS.items())


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    ranked: list[Hypothesis] = []
    for hypothesis in hypotheses:
        hypothesis.weighted_score = compute_weighted_total_score(hypothesis)
        ranked.append(hypothesis)
    return sorted(ranked, key=lambda item: item.weighted_score or 0.0, reverse=True)
