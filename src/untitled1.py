import os

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
AGG_PATH = os.path.join(PROJECT_ROOT, "results", "test", "aggregated_metrics.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "test")

if not os.path.exists(AGG_PATH):
    raise FileNotFoundError(
        f"Metrics file not found: {AGG_PATH}\n"
        "Run: python src/run_robustness.py --data data/test/ --out results/test/ --backends pytesseract"
    )

df = pd.read_csv(AGG_PATH)

if "backend" in df.columns and df["backend"].nunique() > 1:
    df = df.groupby("augmentation").agg({
        "cer_mean": "mean",
        "cer_std": "mean",
        "wer_mean": "mean",
        "wer_std": "mean",
    }).reset_index()

df = df.set_index("augmentation")

fig1, ax1 = plt.subplots(figsize=(12, 8))
df["cer_mean"].sort_values().plot(kind="bar", ax=ax1)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Аугментация")
plt.ylabel("CER среднее по аугментациям")
plt.title("Среднесимвольная ошибка (CER)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cer_plot.png"))
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 8))
df["wer_mean"].sort_values().plot(kind="bar", ax=ax2)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Аугментация")
plt.ylabel("WER среднее по аугментациям")
plt.title("Среднесловарная ошибка (WER)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wer_plot.png"))
plt.show()
