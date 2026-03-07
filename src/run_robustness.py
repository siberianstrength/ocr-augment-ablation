import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import csv
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from augmentations import get_augmentations
from ocr_backends import OCRBackend, get_available_backends
from utils import (
    ExampleVisualization,
    aggregate_metrics,
    cer,
    ensure_dir,
    format_ranking,
    load_image_as_rgb,
    save_before_after_example,
    set_seed,
    wer,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_pairs(data_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:

    image_paths: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, name))

    real: List[Tuple[str, str]] = []
    synthetic: List[Tuple[str, str]] = []

    for img_path in sorted(image_paths):
        base, _ext = os.path.splitext(img_path)
        txt_path = base + ".txt"
        if not os.path.exists(txt_path):
            continue
        pair = (img_path, txt_path)
        fname = os.path.basename(img_path)
        if fname.startswith("synthetic_"):
            synthetic.append(pair)
        else:
            real.append(pair)

    return real, synthetic


def _is_sroie_format(line: str) -> bool:
    parts = line.split(",")
    if len(parts) < 9:
        return False
    try:
        for i in range(8):
            int(parts[i])
        return True
    except ValueError:
        return False


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return ""
    if _is_sroie_format(lines[0]):
        transcripts = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 9:
                transcripts.append(",".join(parts[8:]).strip())
        return " ".join(t for t in transcripts if t)
    return lines[0] if len(lines) == 1 else "\n".join(lines)


def run_single_backend(
    backend: OCRBackend,
    image: np.ndarray,
) -> str:
    if not backend.available:
        raise RuntimeError(f"Backend {backend.name} is not available.")
    return backend.fn(image)


def run_experiment(
    data_dir: str,
    out_dir: str,
    backend_names: Sequence[str],
    seed: int = 42,
    max_images: int = 0,
) -> None:

    set_seed(seed)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "examples"))

    real_pairs, synthetic_pairs = find_image_pairs(data_dir)

    if real_pairs:
        pairs = real_pairs
        used_synthetic = False
    else:
        pairs = synthetic_pairs
        used_synthetic = True

    if not pairs:
        raise RuntimeError(
            f"No (image, txt) pairs found in {data_dir}. "
            "Consider running src/generate_synthetic_data.py first."
        )

    if max_images > 0:
        pairs = pairs[:max_images]

    include_trocr = any(name.lower() == "trocr" for name in backend_names)
    all_backends = get_available_backends(include_trocr=include_trocr)

    selected_backends: Dict[str, OCRBackend] = {}
    for name in backend_names:
        key = name.lower()
        if key not in all_backends:
            print(f"[WARN] Requested backend {name} is not recognized. Available: {list(all_backends.keys())}")
            continue
        backend = all_backends[key]
        if not backend.available:
            print(f"[WARN] Backend {name} is not available: {backend.error_message}")
            continue
        selected_backends[key] = backend

    if not selected_backends:
        raise RuntimeError("No usable OCR backends selected. Check installation of pytesseract / easyocr.")

    augs = get_augmentations()
    rng = np.random.RandomState(seed)

    rows: List[Dict[str, Any]] = []

    examples_to_save = 8
    saved_examples = 0

    start_time = time.time()

    for img_path, txt_path in tqdm(pairs, desc="Images", unit="img"):
        gt_text = load_text(txt_path)
        original = load_image_as_rgb(img_path)
        is_synth = os.path.basename(img_path).startswith("synthetic_")

        for aug in augs:
            aug_img, aug_params = aug.apply(original, rng)
            for backend_name, backend in selected_backends.items():
                try:
                    pred = run_single_backend(backend, aug_img)
                except Exception as exc:
                    print(f"[WARN] Backend {backend.name} failed on {img_path} with aug {aug.name}: {exc}")
                    pred = ""

                cer_val = cer(gt_text, pred)
                wer_val = wer(gt_text, pred)

                rows.append(
                    {
                        "image": os.path.relpath(img_path, data_dir),
                        "augmentation": aug.name,
                        "backend": backend.name,
                        "cer": cer_val,
                        "wer": wer_val,
                        "pred": pred,
                        "gt": gt_text,
                        "augmentation_params": str(aug_params),
                        "is_synthetic": is_synth,
                    }
                )

                if saved_examples < examples_to_save and aug.name != "no_change":
                    example = ExampleVisualization(
                        original=original,
                        augmented=aug_img,
                        gt_text=gt_text,
                        pred_text=pred,
                        backend=backend.name,
                        augmentation_name=aug.name,
                    )
                    out_path = os.path.join(
                        out_dir,
                        "examples",
                        f"example_{saved_examples+1:02d}_{backend.name}_{aug.name}.png",
                    )
                    save_before_after_example(example, out_path)
                    saved_examples += 1

    metrics_path = os.path.join(out_dir, "metrics_summary.csv")
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "augmentation",
                "backend",
                "cer",
                "wer",
                "pred",
                "gt",
                "augmentation_params",
                "is_synthetic",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    aggregated = aggregate_metrics(rows)

    agg_path = os.path.join(out_dir, "aggregated_metrics.csv")
    pd.DataFrame(aggregated).to_csv(agg_path, index=False)

    ranking = format_ranking(aggregated, metric="cer_mean")
    duration_sec = time.time() - start_time

    summary_path = os.path.join(out_dir, "summary.md")
    write_summary_report(
        summary_path=summary_path,
        data_dir=data_dir,
        num_images=len(pairs),
        used_synthetic=used_synthetic,
        backends=list(selected_backends.keys()),
        duration_sec=duration_sec,
        aggregated=aggregated,
        ranking=ranking,
    )

    print(f"Wrote detailed metrics to {metrics_path}")
    print(f"Wrote aggregated metrics to {agg_path}")
    print(f"Wrote summary report to {summary_path}")


def write_summary_report(
    summary_path: str,
    data_dir: str,
    num_images: int,
    used_synthetic: bool,
    backends: List[str],
    duration_sec: float,
    aggregated: List[Dict[str, Any]],
    ranking: Dict[str, List[Tuple[str, float]]],
) -> None:
    ensure_dir(os.path.dirname(summary_path))

    lines: List[str] = []
    lines.append("# OCR robustness with simple augmentations\n")
    lines.append("## Setup\n")
    lines.append(f"- Data directory: `{data_dir}`")
    lines.append(f"- Number of images: **{num_images}**")
    lines.append(f"- Synthetic data only: **{used_synthetic}**")
    lines.append(f"- OCR backends: `{', '.join(backends)}`")
    lines.append(f"- Total runtime: ~{duration_sec:.1f} seconds\n")

    lines.append("## Aggregated metrics\n")
    lines.append("Per-augmentation, per-backend mean CER / WER:\n")

    header = "| Augmentation | Backend | CER mean | CER std | WER mean | WER std | N |"
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for row in sorted(aggregated, key=lambda r: (r["backend"], r["augmentation"])):
        lines.append(
            f"| {row['augmentation']} | {row['backend']} | "
            f"{row['cer_mean']:.3f} | {row['cer_std']:.3f} | "
            f"{row['wer_mean']:.3f} | {row['wer_std']:.3f} | {row['n']} |"
        )
    lines.append("")

    lines.append("## Which augmentations hurt the most?\n")
    lines.append(
        "Lower CER is better. The lists below show augmentations sorted by **mean CER** for each backend."
    )
    lines.append("")
    for backend, items in ranking.items():
        lines.append(f"### Backend: {backend}")
        for aug, value in items:
            if np.isnan(value):
                continue
            lines.append(f"- **{aug}**: CER ≈ {value:.3f}")
        lines.append("")



    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR robustness experiment with augmentations.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data directory (with image + .txt pairs).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for metrics, plots, and examples.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="pytesseract,easyocr",
        help="Comma-separated list of OCR backends to use (pytesseract,easyocr,trocr).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentations and reproducibility.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional maximum number of images to use (0 = all).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    backend_names = [name.strip() for name in args.backends.split(",") if name.strip()]
    run_experiment(
        data_dir=args.data,
        out_dir=args.out,
        backend_names=backend_names,
        seed=args.seed,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()

