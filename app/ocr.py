# app/ocr.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image
import pytesseract


def ocr_image(image_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from an image using Tesseract.
    Returns (text, meta).
    """
    img = Image.open(image_path).convert("RGB")
    text = (pytesseract.image_to_string(img) or "").strip()
    meta: Dict[str, Any] = {"engine": "tesseract"}
    return text, meta
