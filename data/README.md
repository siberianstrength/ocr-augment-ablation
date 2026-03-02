## Data layout

This project expects **paired image + text files** for OCR evaluation and (optionally) training.

### Test set (`data/test/`)

- Images: `.png`, `.jpg`, or `.jpeg`.
- Ground truth text: `.txt` files with the **same basename** as the image.

Example:

```text
data/test/
  img_001.png
  img_001.txt
  receipt_a.jpg
  receipt_a.txt
```

Each `.txt` file must contain a single line with the ground-truth transcription of the corresponding image.

### Training set (`data/train/`, optional)

For the optional CRNN finetune experiment, the training set uses the **same format** as the test set:

- `data/train/<name>.(png|jpg|jpeg)`
- `data/train/<name>.txt`

You can keep this folder empty if you do not plan to run the finetuning experiment.

---

## Synthetic demo test set

If you do not have your own data yet, you can generate a small **synthetic test set**:

```bash
python src/generate_synthetic_data.py --out data/test --num_samples 30
```

This script:

- Creates simple text images using Pillow (with random strings, fonts, and light distortions).
- Saves them as `synthetic_XXXX.png` plus `synthetic_XXXX.txt` in the target folder.
- Marks them as **synthetic** in the logs produced by `src/run_robustness.py`.

When you later add your own **real** data (files whose names **do not** start with `synthetic_`), the robustness script will automatically:

- Prefer real examples and
- **Exclude synthetic** ones from the experiment to avoid mixing them.

This enables a quick demo run while keeping your real evaluation clean.

---

## ICDAR SROIE dataset

If you have the **SROIE** (Scanned Receipts OCR) dataset:

1. **You need both images and annotations.** The folder `0325updated.task1train(626p)` often contains only `.txt` files. The receipt images (`.jpg`) are usually in a separate archive.
2. **Download images** from the [SROIE Google Drive](https://drive.google.com/open?id=1ShItNWXyiYtFDM5W02bceHuJjyeeJl2) and extract them.
3. **Convert to our format**:

```powershell
# If images are in the SAME folder as annotations:
python src\prepare_sroie.py --sroie_dir "data\0325updated.task1train(626p)" --out data\test

# If images are in a DIFFERENT folder (e.g. task1train):
python src\prepare_sroie.py --sroie_dir "data\0325updated.task1train(626p)" --images_dir "data\task1train" --out data\test
```

4. **Run the experiment**:

```powershell
.\.venv\Scripts\python.exe src\run_robustness.py --data data\test --out results --backends pytesseract,easyocr
```

---

## Notes and tips

- For best results, use images that are already cropped roughly to the text region (receipts, forms, scanned pages, etc.).
- If you want more handwriting-like fonts in the synthetic data, place additional `.ttf` font files in a folder of your choice and pass the path via `--fonts_dir` to `generate_synthetic_data.py` (see that script for details).
- Make sure your `.txt` ground-truth files are encoded in UTF-8.

