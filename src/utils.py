# src/utils.py
# Safe utilities for the OCR robustness experiment.
# - avoids importing heavy C-extensions at module import time
# - provides pure-Python CER/WER
# - uses Pillow for image composition and only imports matplotlib lazily

import os
import csv
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------- reproducibility ----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch optional
        pass


# ---------------- Levenshtein-like functions (pure Python) ----------------
def _levenshtein_chars(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    ref = "" if ref is None else str(ref)
    hyp = "" if hyp is None else str(hyp)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = _levenshtein_chars(ref, hyp)
    return float(dist) / float(max(1, len(ref)))


def wer(ref: str, hyp: str) -> float:
    ref_tokens = [] if not ref else str(ref).split()
    hyp_tokens = [] if not hyp else str(hyp).split()
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    m, n = len(ref_tokens), len(hyp_tokens)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    dist = prev[n]
    return float(dist) / float(max(1, m))


# ---------------- I/O helpers ----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_image_as_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image_from_array(path: str, image: np.ndarray) -> None:
    img = Image.fromarray(image.astype(np.uint8))
    ensure_dir(os.path.dirname(path))
    img.save(path)


# ---------------- Example visualization (uses Pillow, no matplotlib) ----------------
@dataclass
class ExampleVisualization:
    original: np.ndarray  # HWC RGB
    augmented: np.ndarray
    gt_text: str
    pred_text: str
    backend: str
    augmentation_name: str


def _draw_multiline_text(img: Image.Image, text: str, position: Tuple[int, int], max_width: int, font: ImageFont.ImageFont):
    # naive wrap
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        wbox = font.getsize(trial)[0]
        if wbox <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = position[1]
    for line in lines:
        img.text((position[0], y), line, fill=(0, 0, 0), font=font)
        y += font.getsize(line)[1] + 2


def save_before_after_example(example: ExampleVisualization, out_path: str, width: int = 800):
    ensure_dir(os.path.dirname(out_path))
    left = Image.fromarray(example.original.astype(np.uint8))
    right = Image.fromarray(example.augmented.astype(np.uint8))
    # Resize to common height
    h = min(left.height, right.height, 600)
    left = left.resize((int(left.width * (h / left.height)), h))
    right = right.resize((int(right.width * (h / right.height)), h))
    padding = 10
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Compose
    text_area_height = 160
    total_w = left.width + right.width + padding * 3
    total_h = h + text_area_height + padding * 2
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    canvas.paste(left, (padding, padding))
    canvas.paste(right, (padding * 2 + left.width, padding))
    draw = ImageDraw.Draw(canvas)
    txt_x = padding
    txt_y = padding + h + 8
    header = f"Aug: {example.augmentation_name} | Backend: {example.backend}"
    draw.text((txt_x, txt_y), header, fill=(0, 0, 0), font=font)
    txt_y += 18
    draw.text((txt_x, txt_y), "GT:", fill=(0, 0, 0), font=font)
    _draw_multiline_text(draw, example.gt_text, (txt_x + 30, txt_y), max_width=total_w - 40, font=font)
    # show pred
    txt_y += 60
    draw.text((txt_x, txt_y), "Pred:", fill=(0, 0, 0), font=font)
    _draw_multiline_text(draw, example.pred_text, (txt_x + 40, txt_y), max_width=total_w - 40, font=font)
    canvas.save(out_path)


# ---------------- Aggregation and plotting (safe) ----------------
def aggregate_metrics(rows: Iterable[Dict[str, Any]], key_fields: Tuple[str, str] = ("augmentation", "backend")) -> List[Dict[str, Any]]:
    # returns list of dicts with fields:
    # augmentation, backend, cer_mean, cer_std, wer_mean, wer_std, n
    from collections import defaultdict

    group = defaultdict(list)
    for r in rows:
        k = (r.get("augmentation"), r.get("backend"))
        group[k].append(r)
    out = []
    for (augmentation, backend), items in sorted(group.items()):
        cer_vals = [float(x.get("cer", 0.0)) for x in items]
        wer_vals = [float(x.get("wer", 0.0)) for x in items]
        n = len(items)
        cer_mean = float(np.mean(cer_vals)) if n > 0 else float("nan")
        cer_std = float(np.std(cer_vals, ddof=1)) if n > 1 else 0.0
        wer_mean = float(np.mean(wer_vals)) if n > 0 else float("nan")
        wer_std = float(np.std(wer_vals, ddof=1)) if n > 1 else 0.0
        out.append(
            {
                "augmentation": augmentation,
                "backend": backend,
                "cer_mean": cer_mean,
                "cer_std": cer_std,
                "wer_mean": wer_mean,
                "wer_std": wer_std,
                "n": n,
            }
        )
    return out


def format_ranking(aggregated: Iterable[Dict[str, Any]], metric: str = "cer_mean"):
    # returns dict: backend -> list of (augmentation, value) sorted ascending (lower is better)
    by_backend = {}
    for row in aggregated:
        b = row["backend"]
        by_backend.setdefault(b, []).append((row["augmentation"], float(row.get(metric, float("nan")))))
    for b, items in by_backend.items():
        items.sort(key=lambda t: (math.inf if math.isnan(t[1]) else t[1]))
        by_backend[b] = items
    return by_backend


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_aggregated_metrics_bar(aggregated: Iterable[Dict[str, Any]], outpath: str, metric: str = "cer_mean", error_metric: str = "cer_std"):
    labels = []
    values = []
    errs = []
    for row in aggregated:
        labels.append(f"{row['augmentation']}")
        values.append(float(row.get(metric, float("nan"))))
        errs.append(float(row.get(error_metric, 0.0)))
    plt = _import_matplotlib()
    ensure_dir(os.path.dirname(outpath))
    if plt is None:
        # fallback: write a small CSV so user can plot locally
        csv_path = outpath.replace(".png", ".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["augmentation", metric, error_metric])
            for l, v, e in zip(labels, values, errs):
                w.writerow([l, v, e])
        return
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, values, yerr=errs, capsize=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)