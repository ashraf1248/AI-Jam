import io
import shutil
import uuid
from pathlib import Path

import pandas as pd
from PIL import Image

from src.pipeline import UploadedArtifact
from src.project_store import ProjectStore
from src.schemas import PipelineInputs, PipelineResult


def _image_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _csv_artifact() -> UploadedArtifact:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    return UploadedArtifact(name="demo.csv", data=frame.to_csv(index=False).encode("utf-8"))


def _workspace_temp_dir() -> Path:
    root = Path("tests") / ".tmp" / f"project-store-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_project_store_round_trip_preserves_result_and_artifacts() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        store = ProjectStore(temp_dir / "saved_projects")
        inputs = PipelineInputs(
            project_title="Saved Demo",
            research_question="What changed between conditions?",
            domain="Biology",
            notes="Resume this later.",
        )
        result = PipelineResult(
            project_title="Saved Demo",
            research_question=inputs.research_question,
            domain=inputs.domain,
            mock_mode=True,
            inputs_analyzed={"pdfs": [], "csvs": ["demo.csv"], "images": ["plot.png"]},
            final_report_markdown="# Saved Demo",
        )
        artifacts = {
            "pdfs": [],
            "csvs": [_csv_artifact()],
            "images": [UploadedArtifact(name="plot.png", data=_image_bytes())],
        }

        saved = store.save_project(inputs=inputs, result=result, artifacts_by_type=artifacts)
        loaded = store.load_project(saved.project_id)

        assert loaded is not None
        assert loaded.project_title == "Saved Demo"
        assert loaded.result is not None
        assert loaded.result.final_report_markdown == "# Saved Demo"

        decoded = store.deserialize_artifacts(loaded.stored_artifacts)
        assert decoded["csvs"][0].name == "demo.csv"
        assert decoded["images"][0].name == "plot.png"
        assert decoded["images"][0].data == artifacts["images"][0].data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_project_store_lists_saved_projects() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        store = ProjectStore(temp_dir / "saved_projects")
        inputs = PipelineInputs(
            project_title="History Item",
            research_question="Can we reopen this later?",
            domain="Chemistry",
        )

        store.save_project(
            inputs=inputs,
            result=None,
            artifacts_by_type={"pdfs": [], "csvs": [], "images": []},
        )
        summaries = store.list_projects()

        assert len(summaries) == 1
        assert summaries[0].project_title == "History Item"
        assert summaries[0].has_result is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
