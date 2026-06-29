from __future__ import annotations

import json
from typing import Any


def _json_schema_block(example: dict[str, Any]) -> str:
    return json.dumps(example, indent=2)


def literature_evidence_prompt(chunk: str, source_filename: str, research_question: str) -> list[dict[str, str]]:
    example = {
        "evidence_items": [
            {
                "claim": "Concise claim from the paper chunk",
                "mechanism": "Possible mechanism",
                "variables": ["variable_a", "variable_b"],
                "method": "Method used",
                "limitation": "Key limitation or uncertainty",
                "citation": source_filename,
                "confidence": 0.61,
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You extract cautious scientific evidence. Return only JSON. "
                "Do not overclaim. Cite the source filename."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {research_question}\n"
                f"Source filename: {source_filename}\n"
                "Extract 1-3 evidence items from the chunk below. Mark uncertainty clearly.\n"
                f"Expected JSON:\n{_json_schema_block(example)}\n\n"
                f"Chunk:\n{chunk}"
            ),
        },
    ]


def data_interpretation_prompt(dataset_summary: dict[str, Any], research_question: str) -> list[dict[str, str]]:
    example = {"plain_english_summary": "Concise interpretation of patterns and cautions."}
    return [
        {
            "role": "system",
            "content": "You summarize exploratory scientific datasets carefully and only in JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Research question: {research_question}\n"
                f"Dataset profile: {json.dumps(dataset_summary)}\n"
                "Summarize the most relevant patterns, limitations, and caveats.\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]


def image_interpretation_prompt(filename: str, research_question: str) -> str:
    example = {
        "filename": filename,
        "image_type": "microscopy image",
        "visible_patterns": ["pattern 1", "pattern 2"],
        "possible_measurements": ["measurement 1"],
        "uncertainty": "Brief uncertainty statement",
    }
    return (
        f"Research question: {research_question}\n"
        f"Filename: {filename}\n"
        "Describe visible scientific patterns cautiously and return only JSON.\n"
        f"Expected JSON:\n{_json_schema_block(example)}"
    )


def gap_detection_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    example = {
        "knowledge_gaps": [
            {
                "gap_title": "Unresolved mechanism under nutrient stress",
                "description": "What is unknown",
                "why_it_matters": "Why it matters",
                "supporting_evidence": ["Source A", "Dataset B"],
                "missing_information": "Specific unknown data",
                "suggested_next_step": "Practical next step",
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": "You identify scientific knowledge gaps from evidence and return only JSON.",
        },
        {
            "role": "user",
            "content": (
                "Use the evidence, dataset summaries, and image observations to propose 2-6 gaps. "
                "Each gap must explain uncertainty and avoid pretending certainty.\n"
                f"Context: {json.dumps(context)}\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]


def hypothesis_generation_prompt(context: dict[str, Any], num_hypotheses: int) -> list[dict[str, str]]:
    example = {
        "hypotheses": [
            {
                "id": "H-1",
                "hypothesis": "A testable hypothesis",
                "scientific_rationale": "Why it might be true",
                "evidence_for": ["support 1"],
                "evidence_against": ["counterpoint 1"],
                "novelty_claim": "Why it may be novel",
                "testable_prediction": "Prediction",
                "proposed_experiment": "Experiment outline",
                "required_measurements": ["measurement"],
                "controls": ["control"],
                "expected_result": "Expected result",
                "falsification_criteria": "How it can fail",
                "risks_or_limitations": "Main risks",
                "confidence_score": 0.52,
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": "You propose scientific hypotheses carefully and return only JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {num_hypotheses} novel but testable hypotheses from the context below. "
                "Include supporting and opposing evidence. Mark uncertainty explicitly.\n"
                f"Context: {json.dumps(context)}\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]


def hypothesis_critique_prompt(hypothesis: dict[str, Any], context: dict[str, Any]) -> list[dict[str, str]]:
    example = {
        "groundedness": 4,
        "novelty": 3,
        "testability": 4,
        "specificity": 4,
        "plausibility": 3,
        "usefulness": 4,
        "main_weakness": "Weakness",
        "suggested_revision": "Revision",
        "should_keep": True,
    }
    return [
        {
            "role": "system",
            "content": "You are a skeptical scientific critic. Return only JSON with 1-5 scores.",
        },
        {
            "role": "user",
            "content": (
                f"Context: {json.dumps(context)}\n"
                f"Hypothesis: {json.dumps(hypothesis)}\n"
                "Score the hypothesis for groundedness, novelty, testability, specificity, plausibility, and usefulness. "
                "Keep the critique constructive and cautious.\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]


def experiment_design_prompt(hypothesis: dict[str, Any], context: dict[str, Any]) -> list[dict[str, str]]:
    example = {
        "objective": "Objective",
        "variables": ["independent", "dependent"],
        "controls": ["control"],
        "procedure": ["step 1", "step 2"],
        "measurements": ["measurement"],
        "expected_outcomes": "Expected outcomes",
        "failure_modes": ["failure mode"],
        "approximate_feasibility_level": "medium",
        "ethical_or_safety_notes": "Notes",
    }
    return [
        {
            "role": "system",
            "content": "You design practical experiments and return only JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Context: {json.dumps(context)}\n"
                f"Hypothesis: {json.dumps(hypothesis)}\n"
                "Design a practical experiment with cautious scientific language.\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]


def final_report_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    example = {
        "evidence_summary": "Summary",
        "dataset_summary": "Summary",
        "image_summary": "Summary",
        "limitations": ["limitation"],
        "failure_modes": ["failure mode"],
        "recommended_next_steps": ["next step"],
    }
    return [
        {
            "role": "system",
            "content": "You draft concise scientific reports and return only JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Context: {json.dumps(context)}\n"
                "Summarize the project into report-ready sections with explicit limitations.\n"
                f"Expected JSON:\n{_json_schema_block(example)}"
            ),
        },
    ]
