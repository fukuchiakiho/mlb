import sys
sys.path.append('..')
import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, LSTM
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

# ファイルパスと名前を設定
name1 = "20220407_1005_learn"#2022Ver
#name1 = "20220407_1005_test"#2022Ver
#name1 = "20230330_1004_learn"#2023Ver
name2 = "test"
path1 = os.path.join('/home/fkch/mlb/data/vecdata/' + name1 + '.csv')
path2 = os.path.join('/home/fkch/mlb/' + name2 + '.csv')

# CSVファイルを読み込む
reader1 = pd.read_csv(path1, sep=',')
reader2 = pd.read_csv(path2, sep=',')

# データを結合する
df = pd.merge(reader1, reader2, on='play', how='left')

# 欠損値を補完する
df_x = df.fillna({'ID': 15})

# 変数の初期化
m = 0
z = 1
max_mask = []
max_counter = []
epo = 1000
year = 2022

def split_games(x):
    '''
    試合を分割するための関数
    inn(x)の値が小さくなったらnumber(z)を加算してreturnする
    '''
    global m, z
    if x > m:
        m += 1
        return z
    elif x == m:
        return z
    elif x < m:
        m = -1
        z += 1
        return z
    else:
        return "miss"

# 試合を分割する
df_x["number"] = df_x['Inn.'].apply(split_games)
max_num = df_x['number'].max()

# リストの初期化
m = []
tm = []
l = []
tl = []
x = []
tx = []
t = []
all_l = []
all_tl = []
max_il = []
sequences = []

# 入力データの作成
for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j].copy()
    new_nm = df_x[df_x['number'] == j].copy()
    target_string = 'stealing'
    nm = nm[~nm['play'].str.contains(target_string)]
    nm.drop_duplicates(subset=['Player'], inplace=True)
    #p_nm = nm[nm['Player']].copy()
    inn_9 = nm[:9]
    score_9 = nm.tail(1)
    for index, row in inn_9.iterrows():
        # 特徴量を取得
        x.append(row['HR'])
        x.append(row['R'])
        x.append(row['RBI'])
        x.append(row['SB'])
        #x.append(row['BB%'])
        #x.append(row['K%'])
        x.append(row['ISO'])
        x.append(row['BABIP'])
        x.append(row['AVG'])
        x.append(row['OBP'])
        x.append(row['SLG'])
        #x.append(row['wOBA'])
        #x.append(row['wRC+'])
        x.append(row['BsR'])
        x.append(row['Off'])
        x.append(row['Def'])
        x.append(row['WAR'])
        #tx.append(row['ID'])
        #x = row['WAR']
        m.append(x)
        x = []
    for index, row in new_nm.iterrows():
        # 目的変数を取得
        tx = row['Score']
    tm.append(tx)
    sequences.append(m)
    m = []

# ndarray形式に変換
sequences = np.array(sequences)
scores = np.array(tm)
scores = scores.astype('int64')

# Bidirectionalモデルの構築
hidden_units = 6   # Bidirectionalの隠れユニット数
model = Sequential()
#model.add(Bidirectional(LSTM(hidden_units)))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(Bidirectional(LSTM(3)))
model.add(Dense(1, activation='relu'))

# モデルのコンパイル
model.compile(loss='mean_squared_error', optimizer='adam')

# モデルの学習
history = model.fit(sequences, scores, epochs=epo, batch_size=10)

# モデルの保存
pickle.dump(model, open(f'Bidirectional_{epo}_{year}.pkl', 'wb'))

# 学習曲線のプロット
mse = history.history['loss']
rmse = np.sqrt(mse)
np.save(f'rmse_{epo}_{year}.npy', rmse)
plt.plot(rmse)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig(f'Bidirectional_{epo}_{year}.png')
plt.savefig(f'Bidirectional_{epo}_{year}.eps')