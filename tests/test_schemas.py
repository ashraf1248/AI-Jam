from src.schemas import ExperimentPlan, Hypothesis, HypothesisCritique


def test_hypothesis_schema_validation() -> None:
    hypothesis = Hypothesis.model_validate(
        {
            "id": "H-1",
            "hypothesis": "Increasing nutrient pulsing may shift response dynamics.",
            "scientific_rationale": "Uploaded evidence points to condition-sensitive dynamics.",
            "evidence_for": ["Paper A suggests stress adaptation."],
            "evidence_against": ["Dataset size is small."],
            "novelty_claim": "Combines two underlinked signals.",
            "testable_prediction": "The treated group will diverge after pulsing.",
            "proposed_experiment": "Run a pulsing pilot.",
            "required_measurements": ["growth", "marker abundance"],
            "controls": ["baseline"],
            "expected_result": "A moderate but repeatable shift appears.",
            "falsification_criteria": "No shift remains after replication.",
            "risks_or_limitations": "Potential confounding by batch effects.",
            "confidence_score": 0.58,
            "critique": {
                "groundedness": 4,
                "novelty": 3,
                "testability": 4,
                "specificity": 4,
                "plausibility": 3,
                "usefulness": 4,
                "main_weakness": "Needs tighter endpoint definition.",
                "suggested_revision": "Specify one primary endpoint.",
                "should_keep": True,
            },
        }
    )
    assert isinstance(hypothesis.critique, HypothesisCritique)
    assert hypothesis.id == "H-1"


def test_experiment_plan_flattens_nested_string_fields() -> None:
    plan = ExperimentPlan.model_validate(
        {
            "hypothesis_id": "H-1",
            "objective": "Discriminate between crosslinking mechanisms.",
            "variables": [
                {"independent": ["citric acid concentration", "curing humidity"]},
                {"dependent": "FTIR ester peak"},
            ],
            "controls": {"baseline": ["no crosslinker"]},
            "procedure": [
                "Prepare matched film batches.",
                {"measure": ["FTIR", "water uptake"]},
            ],
            "measurements": {"primary": ["tensile strength", "elongation at break"]},
            "expected_outcomes": "Crosslinked films should show stronger ester signatures.",
            "failure_modes": [{"risk": "phase separation"}],
            "approximate_feasibility_level": "medium",
            "ethical_or_safety_notes": "Follow standard lab PPE requirements.",
        }
    )

    assert "independent: citric acid concentration, curing humidity" in plan.variables
    assert plan.controls == ["baseline: no crosslinker"]
    assert "measure: FTIR, water uptake" in plan.procedure
