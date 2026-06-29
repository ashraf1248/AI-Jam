from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def safe_json_loads(text: str) -> Any:
    return json.loads(text)


def repair_json_string(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "{}"
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    candidate = match.group(1) if match else stripped
    candidate = candidate.replace("```json", "").replace("```", "").strip()
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return candidate


def parse_json_with_repair(text: str) -> Any:
    try:
        return safe_json_loads(text)
    except json.JSONDecodeError:
        repaired = repair_json_string(text)
        return safe_json_loads(repaired)


def preview_list(items: Iterable[str], limit: int = 3) -> list[str]:
    values = [item for item in items if item]
    return values[:limit]


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
