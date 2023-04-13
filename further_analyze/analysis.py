import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('delta.csv')
df = df.dropna()
grouped_df = df.groupby("word_t")
title_list = ["Vision", "Motor", "Socialness", "Emotion_abs+1", "Time", "Space"]
for group_name, group in grouped_df:
    for i in range(6):
        sorted_group = group.sort_values(title_list[i], key=lambda x: x.abs(), ascending=False)
        plt.figure(figsize=(20, 10))
        bar = plt.bar(sorted_group['word_t1'], sorted_group[title_list[i]])
        # print(f"Group: {group_name}")
        # print(sorted_group)