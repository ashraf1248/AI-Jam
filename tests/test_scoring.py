from src.schemas import Hypothesis, HypothesisCritique, RefinedHypothesis
from src.scoring import rank_hypotheses, score_refined_hypotheses


def _make_hypothesis(identifier: str, groundedness: int, novelty: int, testability: int) -> Hypothesis:
    return Hypothesis(
        id=identifier,
        hypothesis=f"Hypothesis {identifier}",
        scientific_rationale="Rationale",
        evidence_for=["support"],
        evidence_against=["counter"],
        novelty_claim="Novelty",
        testable_prediction="Prediction",
        proposed_experiment="Experiment",
        required_measurements=["measurement"],
        controls=["control"],
        expected_result="Expected",
        falsification_criteria="Falsify",
        risks_or_limitations="Limitations",
        confidence_score=0.5,
        critique=HypothesisCritique(
            groundedness=groundedness,
            novelty=novelty,
            testability=testability,
            specificity=4,
            plausibility=3,
            usefulness=4,
            main_weakness="Weakness",
            suggested_revision="Revision",
            should_keep=True,
        ),
    )


def test_rank_hypotheses_orders_by_weighted_score() -> None:
    lower = _make_hypothesis("H-1", groundedness=3, novelty=2, testability=3)
    higher = _make_hypothesis("H-2", groundedness=5, novelty=4, testability=5)
    ranked = rank_hypotheses([lower, higher])
    assert ranked[0].id == "H-2"
    assert ranked[0].weighted_score > ranked[1].weighted_score


def test_score_refined_hypotheses_sets_weighted_score_and_delta() -> None:
    original = _make_hypothesis("H-1", groundedness=3, novelty=3, testability=3)
    original = rank_hypotheses([original])[0]
    refined = RefinedHypothesis(
        hypothesis_id="H-1",
        original_hypothesis="Original",
        improved_hypothesis="Improved",
        why_original_was_insufficient="Too broad",
        hidden_assumption="Single pathway",
        sharper_mechanism="Interaction-sensitive mechanism",
        key_interaction_or_missing_variable="Moderator X",
        revised_prediction="Only subgroup A changes",
        mechanism_discriminating_experiment="Factorial experiment",
        what_result_would_distinguish_mechanisms="Interaction effect",
        why_this_is_more_novel="More discriminating",
        residual_uncertainty="Moderator measurement quality",
        refined_critique=HypothesisCritique(
            groundedness=4,
            novelty=5,
            testability=4,
            specificity=5,
            plausibility=4,
            usefulness=4,
            main_weakness="Still measurement-limited",
            suggested_revision="Tighten assay",
            should_keep=True,
        ),
    )

    scored = score_refined_hypotheses([refined], [original])[0]

    assert scored.weighted_score is not None
    assert scored.score_delta_vs_original is not None
    assert scored.score_delta_vs_original > 0
