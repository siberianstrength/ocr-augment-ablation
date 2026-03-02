## OCR Augmentation Ablation

This repository implements a complete ablation study of how simple image augmentations affect the robustness of modern OCR engines.

The core ideas:

- **Robustness test (required)**: apply a fixed suite of augmentations to a test set of text images and measure the impact on OCR quality (CER/WER) for **Tesseract** (`pytesseract`) and **EasyOCR**.
- **Finetune with augmentation (optional)**: if a GPU is available and the user explicitly enables it, quickly finetune a small CRNN-based OCR model with and without augmentations and compare performance.

The code is designed to run **out of the box on CPU** on a small test set (20–50 images). If Tesseract, EasyOCR, or a GPU are not available, the corresponding backends/features are skipped with a clear warning.

### Repository structure

- **`README.md`**: this file, high-level overview and quickstart.
- **`requirements.txt`**: pinned Python dependencies.
- **`run_experiment.sh`**: convenience wrapper to run robustness or (optionally) finetune experiments from the command line.
- **`data/`**
  - **`README.md`**: how to prepare or replace the dataset.
  - **`test/`**: test images and ground-truth text files.
  - **`train/`** (optional): training set for CRNN finetuning.
- **`notebooks/`**
  - **`01_data_prep.ipynb`**: inspect data, visualize examples, validate CER/WER implementations.
  - **`02_run_robustness.ipynb`**: run the robustness ablation, compute metrics, and plot results.
  - **`03_optional_finetune.ipynb`**: GPU-only quick finetune experiment for a lightweight CRNN OCR model.
- **`src/`**
  - **`utils.py`**: metrics (CER/WER), IO helpers, visualization utilities, seed control, simple logging.
  - **`augmentations.py`**: augmentation definitions using `albumentations` plus some light custom logic.
  - **`ocr_backends.py`**: wrappers around `pytesseract`, `easyocr`, and optional TrOCR (disabled by default).
  - **`run_robustness.py`**: main script for the robustness test experiment.
  - **`generate_synthetic_data.py`**: utility to generate a small synthetic test set when no data is provided.
  - **`finetune_crnn.py`**: simple CRNN model and training loop for optional GPU-only finetuning.
- **`results/`**
  - **`metrics_summary.csv`** (generated): per-image, per-augmentation, per-backend metrics.
  - **`examples/`** (generated): visual “original vs augmented vs prediction” examples.
  - **`summary.md`** (generated): short markdown report with main findings and recommendations.

---

### Installation

1. **Create environment (recommended)**:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Install Tesseract (required for `pytesseract`)**:

- Linux: install via your package manager (e.g. `sudo apt-get install tesseract-ocr`).
- Windows/macOS: download installers from the official Tesseract project.

Make sure the `tesseract` binary is on your `PATH` or configure `pytesseract.pytesseract.tesseract_cmd` manually (see comments in `src/ocr_backends.py`).

> **Note:** EasyOCR and the optional CRNN finetune use PyTorch. If no GPU is available, they will run on CPU but training will be slower; the finetune notebook is designed to **skip training** when no CUDA GPU is detected.

---

### Data format

Expected structure for the **test set**:

- Images in `data/test/` with names like `img_001.png`, `invoice_05.jpg`, etc.
- For each image `<name>.<ext>`, a corresponding text file `<name>.txt` in the same folder containing the **ground-truth text string**.

Supported image formats: **PNG**, **JPG/JPEG**.

Example:

```text
data/test/
  img_001.png
  img_001.txt  # "TOTAL: 123.45 USD"
  receipt_a.jpg
  receipt_a.txt
```

If **no data** is provided in `data/test/`, the pipeline can automatically create a **small synthetic test set** (~20–50 images) under `data/test/` using `src/generate_synthetic_data.py`. These files are prefixed with `synthetic_` and are treated as **synthetic** in the logs. If the user later adds their own (non-`synthetic_`) files, the experiment will automatically **ignore synthetic** examples and only use real ones.

Details and examples are given in `data/README.md`.

---

### Augmentation suite

The robustness experiment uses a fixed set of augmentations with deterministic parameter ranges:

- **`no_change`**: baseline (original image).
- **`rotation`**: random angle \(\in U(-15^\circ, +15^\circ)\).
- **`elastic`**: `ElasticTransform` with `alpha_affine=30`, `sigma=4`.
- **`gaussian_blur`**: Gaussian blur with kernel size 3–7 and \(\sigma \in [0.5, 2.0]\).
- **`jpeg_compression`**: JPEG artifacts with quality in `{30, 50, 80}`.
- **`brightness_contrast`**: contrast scaling in \([0.7, 1.3]\) and brightness shift \(\pm 20\).
- **`salt_and_pepper`**: noise with amount in `[0.01, 0.03]`.
- **`affine`**: scaling in `[0.9, 1.1]` and shear \(\pm 10^\circ\).
- **`combined_mix`**: random combination of 2–3 of the above (stress test).

Each augmentation has a fixed, reproducible implementation (controlled by a global seed) and its **actual sampled parameters** are logged alongside the metrics in `results/metrics_summary.csv`.

---

### Robustness experiment (Experiment 1)

**Goal:** evaluate how each augmentation affects OCR performance.

For each image–text pair in `data/test/` and for each augmentation \(A\):

1. Apply \(A\) to get an augmented image.
2. Run OCR backends:
   - **Tesseract** via `pytesseract`.
   - **EasyOCR**.
3. Compute **CER** and **WER** between prediction and ground truth.
4. Save at least one **“before/after/prediction”** visualization per augmentation to `results/examples/`.
5. Log results to `results/metrics_summary.csv` with columns:
   - `image`, `augmentation`, `backend`, `cer`, `wer`, `pred`
   - plus extra fields such as `gt`, `augmentation_params`, `is_synthetic`.

Additionally, the script:

- Aggregates metrics per `(augmentation, backend)` (mean, std, count).
- Saves a bar plot with error bars to `results/` (PNG).
- Generates a short markdown report `results/summary.md` summarizing:
  - Which augmentations hurt performance the most.
  - How Tesseract vs EasyOCR compare.
  - Recommendations for data pipelines and potential future work.

---

### Optional: Finetune with augmentation (Experiment 2, GPU-only)

If a **CUDA GPU** is available and the user explicitly opts in, you can run a small finetuning experiment using a lightweight **CRNN** model with CTC loss:

- Training set in `data/train/` with the same (`.png`/`.jpg` + `.txt`) format as `data/test/`.
- Compare:
  - **Baseline**: training without augmentations.
  - **Augmented**: training with a subset of the strongest augmentations.
- Train for ~10–20 epochs on up to ~200 examples.
- Evaluate on the **original (non-augmented) test set** and report \(\Delta\) CER.

This workflow is implemented primarily in `notebooks/03_optional_finetune.ipynb`, which uses utilities from `src/finetune_crnn.py`. The notebook **automatically skips training** if no GPU is detected.

---

### Optional TrOCR backend

The file `src/ocr_backends.py` includes **optional support** for a TrOCR backend (Hugging Face Transformers) if:

- `transformers` and its dependencies are installed, **and**
- there is sufficient GPU memory (checked via `torch.cuda.is_available()`).

TrOCR is **disabled by default** and is **not** included in `requirements.txt`. To enable it:

1. Install extra dependencies manually, for example:

```bash
pip install transformers sentencepiece accelerate
```

2. Pass `--backends pytesseract,easyocr,trocr` when running the robustness script.

If the environment cannot load the TrOCR model, the backend is skipped with a warning and does not break the experiment.

---

### Quickstart: run robustness experiment

From the repository root (`ocr-augment-ablation/`):

```bash
# 1) Install dependencies (once)
pip install -r requirements.txt

# 2) Optionally: generate a small synthetic test set
python src/generate_synthetic_data.py --out data/test --num_samples 30

# 3) Run robustness experiment with Tesseract and EasyOCR
bash run_experiment.sh --mode robustness --data data/test --backends pytesseract,easyocr --out results
```

Or call the Python script directly:

```bash
python src/run_robustness.py --data data/test --out results --backends pytesseract,easyocr
```

After completion, inspect:

- `results/metrics_summary.csv` – full log of predictions and metrics.
- `results/summary.md` – short markdown report.
- `results/examples/` – visual examples.

On a typical CPU-only machine, running the robustness pipeline on ~50 images with both backends usually takes **a few minutes** (order of **1–5 minutes**, depending on EasyOCR and Tesseract speed). On a GPU machine, the robustness run is still CPU-bound but the optional finetuning notebook can complete a small 10–20 epoch run in **5–15 minutes**, depending on GPU speed and dataset size.

---

### Notebooks

To work with the notebooks locally:

```bash
jupyter notebook notebooks/01_data_prep.ipynb
```

Each notebook starts with a cell that:

- Installs dependencies via `pip install -r ../requirements.txt` (if needed).
- Checks availability of Tesseract, EasyOCR, and GPU.

You can also upload the notebooks to Google Colab. Make sure to upload the entire repository or mount it via Google Drive so that relative paths (e.g. `../src`) work correctly.

