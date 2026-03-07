import os
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


@dataclass
class ExampleVisualization:
    original: np.ndarray
    augmented: np.ndarray
    gt_text: str
    pred_text: str
    backend: str
    augmentation_name: str


def _draw_multiline_text(draw, text: str, position: Tuple[int, int], max_width: int, font):
    def measure(s: str) -> Tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), s, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                bbox = font.getbbox(s)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                return font.getsize(s)

    words = text.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        wbox, _ = measure(trial)
        if wbox <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    x0, y = position
    for line in lines:
        draw.text((x0, y), line, fill=(0, 0, 0), font=font)
        _, h = measure(line)
        y += int(h + 4)


def save_before_after_example(example: ExampleVisualization, out_path: str, width: int = 800):
    ensure_dir(os.path.dirname(out_path))
    left = Image.fromarray(example.original.astype(np.uint8))
    right = Image.fromarray(example.augmented.astype(np.uint8))
    h = min(left.height, right.height, 600)
    left = left.resize((int(left.width * (h / left.height)), h))
    right = right.resize((int(right.width * (h / right.height)), h))
    padding = 10
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
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
    txt_y += 60
    draw.text((txt_x, txt_y), "Pred:", fill=(0, 0, 0), font=font)
    _draw_multiline_text(draw, example.pred_text, (txt_x + 40, txt_y), max_width=total_w - 40, font=font)
    canvas.save(out_path)


def aggregate_metrics(rows: Iterable[Dict[str, Any]], key_fields: Tuple[str, str] = ("augmentation", "backend")) -> List[Dict[str, Any]]:
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
    by_backend = {}
    for row in aggregated:
        b = row["backend"]
        by_backend.setdefault(b, []).append((row["augmentation"], float(row.get(metric, float("nan")))))
    for b, items in by_backend.items():
        items.sort(key=lambda t: (math.inf if math.isnan(t[1]) else t[1]))
        by_backend[b] = items
    return by_backend

