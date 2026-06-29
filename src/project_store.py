from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import UploadedArtifact
from src.schemas import (
    PipelineInputs,
    PipelineResult,
    SavedProject,
    SavedProjectSummary,
    SavedUploadArtifact,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "hypothesis-forge-project"


class ProjectStore:
    def __init__(self, root: str | Path = "saved_projects"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_path(self, project_id: str) -> Path:
        return self.root / f"{project_id}.json"

    @staticmethod
    def _serialize_artifacts(
        artifacts_by_type: dict[str, list[UploadedArtifact]],
    ) -> dict[str, list[SavedUploadArtifact]]:
        serialized: dict[str, list[SavedUploadArtifact]] = {}
        for artifact_type, artifacts in artifacts_by_type.items():
            serialized[artifact_type] = [
                SavedUploadArtifact(
                    name=artifact.name,
                    data_base64=base64.b64encode(artifact.data).decode("ascii"),
                )
                for artifact in artifacts
            ]
        return serialized

    @staticmethod
    def deserialize_artifacts(
        stored_artifacts: dict[str, list[SavedUploadArtifact]],
    ) -> dict[str, list[UploadedArtifact]]:
        deserialized: dict[str, list[UploadedArtifact]] = {}
        for artifact_type, artifacts in stored_artifacts.items():
            deserialized[artifact_type] = [
                UploadedArtifact(
                    name=artifact.name,
                    data=base64.b64decode(artifact.data_base64.encode("ascii")),
                )
                for artifact in artifacts
            ]
        return deserialized

    def save_project(
        self,
        inputs: PipelineInputs,
        result: PipelineResult | None,
        artifacts_by_type: dict[str, list[UploadedArtifact]],
        project_id: str | None = None,
    ) -> SavedProject:
        existing = self.load_project(project_id) if project_id else None
        now = _utc_now_iso()
        resolved_project_id = project_id or f"{_slugify(inputs.project_title)}-{now.replace(':', '').replace('+00:00', 'z')}"
        upload_manifest = {
            artifact_type: [artifact.name for artifact in artifacts]
            for artifact_type, artifacts in artifacts_by_type.items()
        }
        project = SavedProject(
            project_id=resolved_project_id,
            project_title=inputs.project_title,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            inputs=inputs,
            result=result,
            stored_artifacts=self._serialize_artifacts(artifacts_by_type),
            upload_manifest=upload_manifest,
        )
        self._project_path(resolved_project_id).write_text(
            project.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return project

    def load_project(self, project_id: str | None) -> SavedProject | None:
        if not project_id:
            return None
        path = self._project_path(project_id)
        if not path.exists():
            return None
        return SavedProject.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> list[SavedProjectSummary]:
        summaries: list[SavedProjectSummary] = []
        for path in self.root.glob("*.json"):
            try:
                project = SavedProject.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summaries.append(
                SavedProjectSummary(
                    project_id=project.project_id,
                    project_title=project.project_title,
                    updated_at=project.updated_at,
                    has_result=project.result is not None,
                    upload_manifest=project.upload_manifest,
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)
