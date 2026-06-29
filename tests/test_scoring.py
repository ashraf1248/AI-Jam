from src.schemas import Hypothesis, HypothesisCritique
from src.scoring import rank_hypotheses


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
