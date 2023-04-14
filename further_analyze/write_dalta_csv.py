import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import os


# 获取word在六维上的投影值
def get_dims(word):
    word = word.lower()
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    dims = cursor.execute("SELECT Vision, Motor, Socialness, Emotion_abs, Time, Space FROM bert_English WHERE word = ?",
                          (word,))
    dims = dims.fetchall()
    cursor.close()
    conn.close()
    if len(dims) > 0:
        return dims[0]
    else:
        return ()


# dims: [(word_vision, word_motor, ...),]
# dims[0]: (word_vision, word_motor, ...) tuple

# 将df_single_start更新为新用户的第一个guess
def reset(i):
    global user, secret, indexes, guesses, scores, df_single_start, df_single_end
    user = df['user'][i]
    secret = df['secret'][i]
    indexes = [0]
    guesses = [df['guess'][i]]
    df_single_start = i


# 判断用户guess到target word
def analysis(df, task):
    df = df.sort_values(by=df.columns[0], ignore_index=True)
    for i in range(1, len(df)):
        # print(df['guess'][i])
        if df['guess'][i] == task.split('_')[1]:
            # print(f'User {df["user"][i]} finished the {task} in {i} steps.')
            return i
    return False


def write_data(df, where):
    df = df.sort_values(by=df.columns[0], ignore_index=True)
    global user, secret, indexes, guesses, scores, df_single_start, df_single_end
    data = []
    file_exists = os.path.exists("delta.csv")
    header = ["user", "from", "index", "word_t", "word_t1", "Vision", "Motor", "Socialness", "Emotion_abs+1", "Time", "Space", "score_t", "score_t1"]
    with open("delta.csv", "a", encoding="UTF8", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for i in range(len(df) - 1):
            word_t = df['guess'][i]
            word_t1 = df['guess'][i + 1]
            dims_t = get_dims(word_t)
            dims_t1 = get_dims(word_t1)
            if len(dims_t) > 0 and len(dims_t1) > 0:
                differences = [b - a for b, a in zip(dims_t1, dims_t)]
            else:
                differences = [np.nan] * 6
            new_row = [user, where, i + 1, word_t, word_t1] + differences + [df['score'][i], df['score'][i + 1]]
            data.append(new_row)
        writer.writerows(data)


if __name__ == '__main__':
    tasks = ["chair_president", "light_heavy", "meet_reach"]
    # tasks = ["meet_reach"]
    for task in tasks:
        df = pd.read_csv(f"./2_csv/{task}.csv")
        df = df.sort_values(by=['user'], ignore_index=True)
        user = df['user'][0]
        secret = df['secret'][0]
        indexes = [0]
        guesses = [df['guess'][0]]
        scores = [df['score'][0]]
        df_single_start = 0
        df_single_end = 0
        for i in range(1, len(df)):
            if df['user'][i] != user:
                #  or pd['guess'][i-1] == "giveup":
                df_single_end = i
                if df_single_end - df_single_start > 0:
                    step = analysis(df=df.iloc[df_single_start:df_single_end], task=task)
                    if step:
                        write_data(df=df.iloc[df_single_start:df_single_end], where=task)
                reset(i)
            else:
                indexes.append(indexes[-1] + 1)
                guesses.append(df['guess'][i])
        step = analysis(df=df.iloc[df_single_start:len(df)], task=task)
        if step:
            write_data(df=df.iloc[df_single_start:len(df)], where=task)
