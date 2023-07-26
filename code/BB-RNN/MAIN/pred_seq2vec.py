import sys
sys.path.append('..')
import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

name1 = "20220407_0831_SabrVec_learn"
name2 = "test"
path1 = os.path.join('/home/fkch/mlb/data/vecdata/' + name1 + '.csv')
path2 = os.path.join('/home/fkch/mlb/' + name2 + '.csv')
reader1 = pd.read_csv(path1, sep=',')
reader2 = pd.read_csv(path2, sep=',')
df = pd.merge(reader1, reader2, on='play', how= 'left')
df_x = df.fillna({'ID': 15})
m = 0
z = 1

with open('/home/fkch/mlb/code/BB-RNN/MAIN/seq2vec_2000.pkl', 'rb') as f:
    params = pickle.load(f)

def team(x):
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
    
df_x["number"] = df_x['Inn.'].apply(team)
max_num = df_x['number'].max()

#リスト初期化
m = []
tm = []
x = []
tx = []
sequences = []

#データ校正

for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j].copy()
    new_nm = df_x[df_x['number'] == j].copy()
    nm.drop_duplicates(subset=['Player'], inplace=True)
    inn_9 = nm[:9]
    score_9 = nm.tail(1)
    for index, row in inn_9.iterrows():
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
        m.append(x)
        x = []
    for index, row in new_nm.iterrows():
        tx = row['Score']
    tm.append(tx)
    sequences.append(m)
    m = []

#ndarray形式にキャスト
test_sequences = np.array(sequences)
scores = np.array(tm)
scores = scores.astype('int64')
#print(test_sequences)

# モデルの生成
hidden_units = 6   # LSTMの隠れユニット数

model = Sequential()
model.add(LSTM(hidden_units, init=params))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')
#model.summary()

#予測
#predicted = model.predict(test_sequences[0].reshape(1, 9, 13))
predicted = model.predict(test_sequences)
#結果
print("予測")
print(predicted)
print("正解")
print(scores)
