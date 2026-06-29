from __future__ import annotations

import csv
import io
import json

from src.schemas import PipelineResult


def export_result_json(result: PipelineResult) -> bytes:
    return json.dumps(result.model_dump(), indent=2).encode("utf-8")


def export_hypotheses_csv(result: PipelineResult) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "hypothesis",
            "weighted_score",
            "confidence_score",
            "should_keep",
            "main_weakness",
        ]
    )
    for hypothesis in result.hypotheses:
        writer.writerow(
            [
                hypothesis.id,
                hypothesis.hypothesis,
                hypothesis.weighted_score,
                hypothesis.confidence_score,
                hypothesis.critique.should_keep if hypothesis.critique else "",
                hypothesis.critique.main_weakness if hypothesis.critique else "",
            ]
        )
    return output.getvalue().encode("utf-8")


def export_markdown_report(result: PipelineResult) -> bytes:
    return result.final_report_markdown.encode("utf-8")
