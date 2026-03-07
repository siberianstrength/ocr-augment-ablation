import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/komme/Desktop/uni/Y2025T3/CV/ocr-augment-ablation/results/aggregated_metrics.csv").set_index('augmentation')
df['cer_mean'].sort_values().plot(
    kind='bar',
    figsize=(12,8))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Аугментация')
plt.ylabel('CER среднее по аугментациям')
plt.title('Среднесимвольная ошибка (CER)')


df['wer_mean'].sort_values().plot(
    kind='bar',
    figsize=(12,8))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Аугментация')
plt.ylabel('WER среднее по аугментациям')
plt.title('Среднесловарная ошибка (WER)')