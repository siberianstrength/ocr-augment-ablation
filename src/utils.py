import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import Levenshtein  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch (if available)."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional; ignore if not available or misconfigured.
        pass


def cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER) as Levenshtein distance / len(ref).

    If the reference is empty:
      - return 0.0 if the hypothesis is also empty
      - return 1.0 otherwise
    """

    ref = ref or ""
    hyp = hyp or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    distance = Levenshtein.distance(ref, hyp)
    return float(distance) / float(len(ref))


def wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) using Levenshtein distance over space-separated tokens."""

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    distance = Levenshtein.distance(" ".join(ref_tokens), " ".join(hyp_tokens))
    return float(distance) / float(len(ref_tokens))


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""

    os.makedirs(path, exist_ok=True)


def load_image_as_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB numpy array [H, W, 3] with values in [0, 255]."""

    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image_from_array(path: str, image: np.ndarray) -> None:
    """Save a numpy array image (H, W, 3) to disk."""

    img = Image.fromarray(image.astype(np.uint8))
    img.save(path)


@dataclass
class ExampleVisualization:
    """Container for a before/after OCR visualization example."""

    original: np.ndarray
    augmented: np.ndarray
    gt_text: str
    pred_text: str
    backend: str
    augmentation_name: str


def save_before_after_example(
    example: ExampleVisualization,
    out_path: str,
    figsize: Tuple[int, int] = (10, 4),
) -> None:
    """Save a side-by-side visualization of original vs augmented image with text annotations."""

    ensure_dir(os.path.dirname(out_path))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(example.original)
    axes[0].set_title("Original")

    axes[1].imshow(example.augmented)
    axes[1].set_title(
        f"Aug: {example.augmentation_name}\n"
        f"Backend: {example.backend}\n"
        f"GT: {truncate_text(example.gt_text)}\n"
        f"Pred: {truncate_text(example.pred_text)}"
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate long text for plotting."""

    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save a dictionary as JSON."""

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def aggregate_metrics(
    rows: Iterable[Dict[str, Any]],
    key_fields: Tuple[str, str] = ("augmentation", "backend"),
) -> List[Dict[str, Any]]:
    """Aggregate CER/WER metrics by (augmentation, backend).

    Returns a list of dicts with:
      - augmentation
      - backend
      - cer_mean, cer_std
      - wer_mean, wer_std
      - n (number of examples)
    """

    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row in rows:
        key = (str(row.get(key_fields[0], "")), str(row.get(key_fields[1], "")))
        cer_val = float(row.get("cer", math.nan))
        wer_val = float(row.get("wer", math.nan))
        if key not in buckets:
            buckets[key] = {"cer": [], "wer": []}
        if not math.isnan(cer_val):
            buckets[key]["cer"].append(cer_val)
        if not math.isnan(wer_val):
            buckets[key]["wer"].append(wer_val)

    aggregated: List[Dict[str, Any]] = []
    for (aug, backend), metrics in buckets.items():
        cer_vals = metrics["cer"]
        wer_vals = metrics["wer"]
        n = max(len(cer_vals), len(wer_vals))
        if n == 0:
            continue
        cer_mean = float(np.mean(cer_vals)) if cer_vals else math.nan
        cer_std = float(np.std(cer_vals)) if cer_vals else math.nan
        wer_mean = float(np.mean(wer_vals)) if wer_vals else math.nan
        wer_std = float(np.std(wer_vals)) if wer_vals else math.nan
        aggregated.append(
            {
                "augmentation": aug,
                "backend": backend,
                "cer_mean": cer_mean,
                "cer_std": cer_std,
                "wer_mean": wer_mean,
                "wer_std": wer_std,
                "n": n,
            }
        )
    return aggregated


def plot_aggregated_metrics_bar(
    aggregated: List[Dict[str, Any]],
    out_path: str,
    metric: str = "cer_mean",
    error_metric: str = "cer_std",
) -> None:
    """Create a bar plot with error bars from aggregated metrics."""

    ensure_dir(os.path.dirname(out_path))

    if not aggregated:
        return

    backends = sorted({row["backend"] for row in aggregated})
    augmentations = sorted({row["augmentation"] for row in aggregated})

    x = np.arange(len(augmentations))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, backend in enumerate(backends):
        values = []
        errors = []
        for aug in augmentations:
            row = next(
                (r for r in aggregated if r["backend"] == backend and r["augmentation"] == aug),
                None,
            )
            if row is None:
                values.append(0.0)
                errors.append(0.0)
            else:
                values.append(float(row.get(metric, 0.0)))
                errors.append(float(row.get(error_metric, 0.0)))
        ax.bar(
            x + i * width,
            values,
            width,
            yerr=errors,
            label=backend,
            capsize=3,
        )

    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(augmentations, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("OCR robustness by augmentation and backend")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_ranking(
    aggregated: List[Dict[str, Any]],
    metric: str = "cer_mean",
) -> Dict[str, List[Tuple[str, float]]]:
    """Return a ranking of augmentations by metric (ascending) for each backend."""

    ranking: Dict[str, List[Tuple[str, float]]] = {}
    for row in aggregated:
        backend = str(row["backend"])
        aug = str(row["augmentation"])
        value = float(row.get(metric, math.nan))
        ranking.setdefault(backend, []).append((aug, value))

    for backend, items in ranking.items():
        items.sort(key=lambda x: (math.inf if math.isnan(x[1]) else x[1]))
        ranking[backend] = items

    return ranking

