import numpy as np
import pandas as pd
import sqlite3
from random import sample
import matplotlib.pyplot as plt


def create_table():
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE bert_English")
    cursor.execute('''CREATE TABLE bert_English (word varchar(100), Vision DOUBLE, Motor DOUBLE,
                        Socialness DOUBLE, Emotion DOUBLE, Emotion_abs DOUBLE, Time DOUBLE, Space DOUBLE)''')
    cursor.close()
    conn.close()


def add_data():
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    data = pd.read_csv("Main_data/Estimated_semantic_dimensions_bert_English.csv", keep_default_na=False)
    print(data.shape)
    data = np.array(data).tolist()
    for i in range(len(data)):
        cursor.execute("insert into bert_English values(?,?,?,?,?,?,?,?)",
                       (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7]))
    cursor.close()
    conn.commit()
    conn.close()


def get_nearby():
    conn = sqlite3.connect("word2vec.db")
    cursor = conn.cursor()
    tot = cursor.execute("SELECT word, neighbor FROM nearby WHERE similarity > 0.6 and similarity <= 0.7;")
    tot = tot.fetchall()
    # print(tot)
    cursor.close()
    conn.close()
    if len(tot) > 1000:
        tot = sample(tot, 1000)
    return tot


def cal_mean():
    # vision = [], motor = [], socialness = [], emotion = [], emotion_abs = [], time = [], space = []
    res = [list() for x in range(7)]
    print(res)
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    nearby = get_nearby()
    print(len(nearby))
    for i in range(len(nearby)):
        tmp = cursor.execute("SELECT * FROM bert_English WHERE word = ?", (nearby[i][0],))
        tmp_list = tmp.fetchall()
        tmp1 = cursor.execute("SELECT * FROM bert_English WHERE word = ?", (nearby[i][1],))
        tmp1_list = tmp1.fetchall()
        # print(tmp_list)
        # print(tmp1_list)
        if len(tmp_list) != 0 and len(tmp1_list) != 0:
            for j in range(7):
                res[j].append(abs(tmp_list[0][j + 1] - tmp1_list[0][j + 1]))
        print(i)
    res_array = np.array(res)
    mean_array = res_array.mean(axis=1)
    print(mean_array)


def draw_mean():
    title_list = ["Vision", "Motor", "Socialness", "Emotion", "Time", "Space"]
    name_list = ["0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    num_list = [[1.663, 1.623, 1.634, 1.594, 1.397, 1.045, 0.773, 0.555, 0.011],
                [0.969, 1.026, 1.061, 1.059, 0.917, 0.739, 0.529, 0.434, 0.010],
                [2.169, 1.928, 1.688, 1.381, 1.199, 1.039, 0.874, 0.634, 0.008],
                [1.687, 1.837, 1.807, 1.541, 1.322, 1.070, 0.819, 0.623, 0.009],
                # [1.348, 1.335, 1.261, 1.028, 0.875, 0.696, 0.560, 0.437, 0.010],
                [1.231, 0.778, 0.692, 0.551, 0.533, 0.475, 0.396, 0.335, 0.007],
                [1.494, 1.416, 1.410, 1.137, 0.986, 0.768, 0.655, 0.596, 0.010]]
    plt.figure(figsize=(40, 30))
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        bar = plt.bar(range(len(num_list[i-1])), num_list[i-1], tick_label=name_list, width=0.6)
        plt.bar_label(bar)
        plt.title(title_list[i-1])
        plt.xlabel("similarity")
        plt.ylabel("abs({}_w - {}_n)".format(title_list[i-1], title_list[i-1]))
    plt.show()


def get_dims(word):
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    dims = cursor.execute("SELECT Vision, Motor, Socialness, Emotion_abs, Time, Space FROM bert_English WHERE word = ?", (word, ))
    dims = dims.fetchall()
    cursor.close()
    conn.close()
    return dims


def draw_trajectory(num, trajectory):
    title_list = ["Vision", "Motor", "Socialness", "Emotion_abs+1", "Time", "Space"]
    tra_list = [list() for x in range(6)]
    # trajectory = ["chair", "seat", "couch", "table", "head", "leader", "president"]
    for tra in trajectory:
        dims = get_dims(tra)
        for j in range(6):
            tra_list[j].append(dims[0][j])
    plt.figure(figsize=(30, 20))
    x = np.arange(num)
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.ylim(-1, 13)
        plt.xticks(x)
        plt.plot(x, tra_list[i - 1])
        plt.plot(x, tra_list[i - 1], "o")
        for j in range(num):
            if i == 1:
                if j + 1 < num:
                    if tra_list[i - 1][j + 1] < tra_list[i - 1][j]:
                        plt.text(j - 0.6, tra_list[i - 1][j] + 1, trajectory[j], color="#5f9e6e")
                    else:
                        plt.text(j - 0.6, tra_list[i - 1][j] - 1, trajectory[j], color="#5f9e6e")
                else:
                    plt.text(j - 0.6, tra_list[i - 1][j] - 1, trajectory[j], color="#5f9e6e")

        for j in range(num - 1):
            delta = round(tra_list[i - 1][j + 1] - tra_list[i - 1][j], 1)
            plt.text(j + 0.2, tra_list[i - 1][j] + delta / 2 + 0.3, delta)
        plt.ylabel("score")
        plt.title(title_list[i - 1])
    plt.show()


def coverage():
    conn = sqlite3.connect("word2vec.db")
    cursor = conn.cursor()
    tot = cursor.execute("SELECT word FROM similarity_range")
    tot = tot.fetchall()
    cursor.close()
    conn.close()
    count = 0
    conn = sqlite3.connect("six_dims.db")
    cursor = conn.cursor()
    for i in range(len(tot)):
        cnt = cursor.execute("SELECT COUNT(*) FROM bert_English WHERE word = ?", (tot[i][0], ))
        cnt = cnt.fetchall()
        print(i, cnt[0][0])
        if cnt[0][0] > 0:
            count = count + 1
    print(count)


if __name__ == "__main__":
    # cal_mean()
    # draw_mean()
    trajectories = [["chair", "seat", "couch", "table", "head", "leader", "president"],
                    ["chair", "table", "seat", "log", "sit", "stool", "swing", "sofa", "bench", "cushion", "chairperson", "chairman", "president"],
                    ["chair", "furniture", "sit", "table", "wooden", "sofa", "cushion", "wood", "comfort", "plastic", "house", "room", "office", "seat", "chair"]]
    draw_trajectory(15, trajectories[2])
    # coverage()


# [1.66290636 0.96940757 2.16862618 1.68737211 1.34800054 1.23146875 1.493855] 0.1 < <= 0.2
# [1.62336817 1.02586414 1.92783491 1.83742164 1.33504953 0.77773788 1.4158963] 0.2 < <= 0.3
# [1.6341573  1.06088102 1.68809323 1.80651891 1.2605146  0.69201143 1.40993432] 0.3 < <= 0.4
# [1.59373275 1.05898119 1.38065516 1.541221   1.02822357 0.55146746 1.13691071] 0.4 < <= 0.5
# [1.39694505 0.91692154 1.19866072 1.32243325 0.87472636 0.53333875 0.98560238] 0.5 < <= 0.6
# [1.04546648 0.73935973 1.03866652 1.0702623  0.69572767 0.47526409 0.76787595] 0.6 < <= 0.7
# [0.77290331 0.52948412 0.87449003 0.81895079 0.5599095  0.39615569 0.65477845] 0.7 < <= 0.8
# [0.55543811 0.43353304 0.6341871  0.68273196 0.43748891 0.33451696 0.59569614] 0.8 < <= 0.9
# [0.01088587 0.00965755 0.00832434 0.00891635 0.0104287  0.00676531 0.00999445] 0.9 < <= 1.0

