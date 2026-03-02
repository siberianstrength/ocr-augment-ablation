from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


class OCRError(RuntimeError):
    """Raised when an OCR backend fails irrecoverably."""


@dataclass
class OCRBackend:
    """Wrapper for a single OCR backend."""

    name: str
    fn: Callable[[np.ndarray], str]
    available: bool
    error_message: Optional[str] = None


def _init_pytesseract_backend() -> OCRBackend:
    try:
        import pytesseract  # type: ignore
        from PIL import Image

        def run(image: np.ndarray) -> str:
            img = Image.fromarray(image)
            # Adjust configuration here if you need to set a custom tesseract_cmd.
            return pytesseract.image_to_string(img, lang="eng").strip()

        return OCRBackend(name="pytesseract", fn=run, available=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        return OCRBackend(
            name="pytesseract",
            fn=lambda _: "",
            available=False,
            error_message=f"pytesseract not available or Tesseract missing: {exc}",
        )


def _init_easyocr_backend() -> OCRBackend:
    try:
        import cv2
        import easyocr  # type: ignore
        import torch

        gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        reader = easyocr.Reader(["en"], gpu=gpu)

        MAX_SIDE = 1600  # Resize large images to avoid EasyOCR OOM on CPU

        def run(image: np.ndarray) -> str:
            h, w = image.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            result = reader.readtext(image, detail=0, paragraph=True)
            return " ".join(result).strip()

        return OCRBackend(name="easyocr", fn=run, available=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        return OCRBackend(
            name="easyocr",
            fn=lambda _: "",
            available=False,
            error_message=f"EasyOCR not available: {exc}",
        )


def _init_trocr_backend() -> OCRBackend:
    """Optional TrOCR backend via Hugging Face transformers.

    This backend is disabled by default and only becomes available if:
      - transformers is installed, and
      - a GPU is available (torch.cuda.is_available()).
    """

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TrOCR backend requires a GPU (CUDA not available).")

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

        device = torch.device("cuda")
        model_name = "microsoft/trocr-base-printed"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        model.eval()

        from PIL import Image

        def run(image: np.ndarray) -> str:
            img = Image.fromarray(image)
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()

        return OCRBackend(name="trocr", fn=run, available=True)
    except Exception as exc:  # pragma: no cover - optional / heavy dependency
        return OCRBackend(
            name="trocr",
            fn=lambda _: "",
            available=False,
            error_message=(
                "TrOCR backend not available. Install transformers and ensure a GPU is present. "
                f"Details: {exc}"
            ),
        )


def get_available_backends(include_trocr: bool = False) -> Dict[str, OCRBackend]:
    """Return a mapping from backend name to OCRBackend instance.

    Parameters
    ----------
    include_trocr:
        If True, attempt to load the optional TrOCR backend (requires transformers and GPU).
    """

    backends: Dict[str, OCRBackend] = {}

    pt = _init_pytesseract_backend()
    backends[pt.name] = pt

    eo = _init_easyocr_backend()
    backends[eo.name] = eo

    if include_trocr:
        trocr = _init_trocr_backend()
        backends[trocr.name] = trocr

    return backends

