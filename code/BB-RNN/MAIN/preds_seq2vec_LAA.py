import sys
sys.path.append('..')
import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import mean_squared_error
import random

name1 = "20220407_0831_SabrVec_learn"
name2 = "test"
path1 = os.path.join('/home/fkch/mlb/data/vecdata/' + name1 + '.csv')
path2 = os.path.join('/home/fkch/mlb/' + name2 + '.csv')
reader1 = pd.read_csv(path1, sep=',')
reader2 = pd.read_csv(path2, sep=',')
df = pd.merge(reader1, reader2, on='play', how= 'left')
df_x = df.fillna({'ID': 15})
p = 0
z = 1
teams = ["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET", "HOU", 
         "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", 
         "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN"]
'''
with open('/home/fkch/mlb/code/BB-RNN/MAIN/seq2vec_2000.pkl', 'rb') as f:
    params = pickle.load(f)'''

loaded_rf_model = pickle.load(open('/home/fkch/mlb/code/BB-RNN/MAIN/seq2vec_2000_LAA.pkl', 'rb'))

def team(x):
    global p, z
    if x > p:
        p += 1
        return z
    elif x == p:
        return z
    elif x < p:
        p = -1
        z += 1
        return z
    else:
        return "miss"
    
df_x = df_x[df_x["Team"] == "LAA"]
df_x["number"] = df_x['Inn.'].apply(team)
df_x.to_csv("LAA_def.csv")
max_num = df_x['number'].max()

#リスト初期化
m = []
tm = []
x = []
tx = []
sequences = []
rmse_list = []
gosa_np = np.empty(0)
gosa_yerr = np.empty(0)
gosa_avg = np.empty(0)
np_avg = np.empty(0)
gosa_max = np.empty(0)
gosa_min = np.empty(0)

#データ校正

for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j].copy()
    new_nm = df_x[df_x['number'] == j].copy()
    nm.drop_duplicates(subset=['Player'], inplace=True)
    inn_9 = nm[:9]
    if len(inn_9) >= 9:
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
        #print(len(m))
        sequences.append(m)
        m = []
    else:
        pass

#ndarray形式にキャスト
test_sequences = np.array(sequences)
scores = np.array(tm)
scores = scores.astype('int64')
#print(test_sequences)

# モデルの生成
hidden_units = 6   # LSTMの隠れユニット数

model = Sequential()
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')
#model.get_weights
#model.summary()
#LAA_seq = test_sequences[0]
#Aプラン　出塁率降順
#print(test_sequences[0])
#test_sequences[0] = test_sequences[0][[0, 1, 3, 6, 5, 2, 4, 8, 7]]
#print(test_sequences[0])
#Bプラン　出塁率昇順
#test_sequences[0] = test_sequences[0][[7, 8, 4, 2, 5, 6, 3, 1, 0]]
#Cプラン　最強選手
test_sequences[0] = test_sequences[0][[0, 0, 0, 0, 0, 0, 0, 0, 0]]
#Dプラン　ランダム
#r_seq = random.sample(range(9), 9)
#print(r_seq)
#test_sequences[0] = test_sequences[0][r_seq]
#predicted = model.predict(test_sequences[0].reshape(1, 9, 13))
y_pred = loaded_rf_model.predict(test_sequences)
y_pred = y_pred.reshape(-1)
#a = loaded_rf_model.get_weights()
#sss = np.copysign(np.ceil(np.abs(y_pred)), y_pred)
#r2_score(y_test, y_pred)
#predicted = model.predict(test_sequences)
#結果

print("予測")
print(y_pred)

print("正解")
print(scores)
'''
print("----")
#for i in range(len(y_pred)):
#gosa = scores[0] - y_pred[0]
rmse = np.sqrt(mean_squared_error(scores, y_pred))
#rmse_list.append(gosa)
print(rmse)

for j in range(len(y_pred)):
    gosa = scores[j] - y_pred[j]
    gosa = abs(gosa)
    gosa_np = np.append(gosa_np, gosa)
gosa_hani = gosa_np.max() - gosa_np.min()
gosa_max = np.append(gosa_max, gosa_np.max())
gosa_min = np.append(gosa_min, gosa_np.min())
np_avg = np.mean(gosa_np)
print(gosa_max)
print(gosa_min)
print(np_avg)
'''