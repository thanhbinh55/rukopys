"""Shared Recognizer V2 contracts for training and inference."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from typing import Iterable


SCORABLE_TYPES = ("handwritten", "printed", "formula", "table", "annotation")
NATURAL_TEXT_TYPES = ("handwritten", "printed", "annotation")

_COMMON_RULES = (
    "Output only the transcription, with no explanation or commentary. "
    "Preserve visible spelling, punctuation, numbers, and symbols. "
    "Do not invent missing content and do not repeat phrases."
)

SYSTEM_PROMPTS = {
    "handwritten": (
        "You are an exact OCR system for handwritten Ukrainian and mixed-script "
        f"document text. {_COMMON_RULES} Use Cyrillic rather than Latin "
        "lookalikes when the writing is Ukrainian."
    ),
    "printed": (
        "You are an exact OCR system for printed document text, including "
        f"Ukrainian, Russian, Latin text, dates, and scanner marks. {_COMMON_RULES}"
    ),
    "formula": (
        "You are an exact OCR system for handwritten or printed mathematical "
        "and scientific formulas. "
        f"{_COMMON_RULES} Preserve operators, subscripts, superscripts, brackets, "
        "fractions, arrows, and line breaks. Plain Unicode or compact LaTeX is "
        "acceptable; never replace a formula with prose."
    ),
    "table": (
        "You are an exact OCR system for document tables. "
        f"{_COMMON_RULES} Return one table row per line and separate visible "
        "columns with |. Preserve empty cells when they are visible."
    ),
    "annotation": (
        "You are an exact OCR system for short handwritten or printed marginal "
        f"annotations, labels, dates, and marks. {_COMMON_RULES}"
    ),
}

USER_PROMPTS = {
    "handwritten": "Transcribe this handwritten region exactly:",
    "printed": "Transcribe this printed region exactly:",
    "formula": "Transcribe this formula exactly:",
    "table": "Transcribe this table exactly:",
    "annotation": "Transcribe this annotation exactly:",
}

DEFAULT_MAX_NEW_TOKENS = {
    "handwritten": 128,
    "printed": 160,
    "formula": 256,
    "table": 384,
    "annotation": 96,
}

_STRIKETHROUGH_CORRECTION = re.compile(r"~~.*?~~\{(.*?)\}")
_STRIKETHROUGH = re.compile(r"~~(.*?)~~")
_INLINE_SPACE = re.compile(r"[ \t\u00A0\u2000-\u200B\u3000]+")
_ALL_SPACE = re.compile(r"\s+")


def canonical_region_type(region_type: str | None) -> str:
    value = str(region_type or "handwritten").lower()
    return value if value in SCORABLE_TYPES else "handwritten"


def prompts_for_type(region_type: str | None) -> tuple[str, str]:
    region_type = canonical_region_type(region_type)
    return SYSTEM_PROMPTS[region_type], USER_PROMPTS[region_type]


def normalize_ocr_target(text: str, region_type: str = "handwritten") -> str:
    """Normalize labels without destroying formula/table row structure."""
    region_type = canonical_region_type(region_type)
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _STRIKETHROUGH_CORRECTION.sub(r"\1", text)
    text = _STRIKETHROUGH.sub(r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if region_type in ("formula", "table"):
        lines = []
        for line in text.split("\n"):
            normalized = _INLINE_SPACE.sub(" ", line).strip()
            if region_type == "table":
                normalized = re.sub(r"\s*\|\s*", "|", normalized).strip("|")
            if normalized:
                lines.append(normalized)
        return "\n".join(lines)

    return _ALL_SPACE.sub(" ", text).strip()


def resize_crop_image(image, max_height: int = 128, min_width: int = 32,
                      max_width: int = 1920):
    """Apply the same image-only resize in training and inference."""
    from PIL import Image

    image = image.convert("RGB")
    if image.height <= max_height:
        return image
    ratio = max_height / max(1, image.height)
    width = max(min_width, min(max_width, int(round(image.width * ratio))))
    return image.resize((width, max_height), Image.Resampling.LANCZOS)


def crop_region(page, bbox, pad: int = 4, jitter: int = 0, rng=None):
    """Crop a region with optional training-only coordinate jitter."""
    from PIL import Image

    page = page.convert("RGB")
    width, height = page.size
    x1, y1, x2, y2 = [int(value) for value in bbox]
    if jitter > 0 and rng is not None:
        x1 += rng.randint(-jitter, jitter)
        y1 += rng.randint(-jitter, jitter)
        x2 += rng.randint(-jitter, jitter)
        y2 += rng.randint(-jitter, jitter)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return Image.new("RGB", (64, 32), "white")
    return resize_crop_image(page.crop((x1, y1, x2, y2)))


def generation_max_new_tokens(region_type: str | None) -> int:
    region_type = canonical_region_type(region_type)
    env_name = f"MAX_TOKENS_{region_type.upper()}"
    if env_name in os.environ:
        return int(os.environ[env_name])
    return DEFAULT_MAX_NEW_TOKENS[region_type]


def generation_kwargs(region_type: str | None) -> dict:
    """Return conservative deterministic generation settings by region type."""
    region_type = canonical_region_type(region_type)
    result = {
        "max_new_tokens": generation_max_new_tokens(region_type),
        "do_sample": False,
        "use_cache": True,
    }
    if region_type in NATURAL_TEXT_TYPES:
        penalty = float(os.getenv("NATURAL_REPETITION_PENALTY", "1.05"))
        if penalty != 1.0:
            result["repetition_penalty"] = penalty
        ngram = int(os.getenv("NATURAL_NO_REPEAT_NGRAM", "0"))
        if ngram > 0:
            result["no_repeat_ngram_size"] = ngram
    return result


def type_balance_multipliers(
    region_types: Iterable[str],
    cap: float = 4.0,
) -> tuple[dict[str, float], Counter]:
    """Capped square-root inverse-frequency weights for gold sampling."""
    normalized = [canonical_region_type(value) for value in region_types]
    counts = Counter(normalized)
    if not counts:
        return {}, counts
    majority = max(counts.values())
    multipliers = {
        region_type: min(cap, math.sqrt(majority / max(count, 1)))
        for region_type, count in counts.items()
    }
    return multipliers, counts
