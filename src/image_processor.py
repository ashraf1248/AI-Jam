from __future__ import annotations

import io

from PIL import Image

from src.schemas import ImageObservation


def load_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    image.load()
    return image


def fallback_image_observation(file_bytes: bytes, filename: str, skipped: bool) -> ImageObservation:
    image = load_image(file_bytes)
    width, height = image.size
    mode = image.mode
    return ImageObservation(
        filename=filename,
        image_type=f"{image.format or 'unknown'} image",
        visible_patterns=[
            f"Resolution is {width}x{height}",
            f"Color mode is {mode}",
        ],
        possible_measurements=["pixel intensity", "region counts", "relative area"],
        uncertainty=(
            "Vision model not configured; these observations come from basic image metadata only."
            if skipped
            else "Observation derived from image metadata and may miss scientific context."
        ),
        skipped=skipped,
    )
