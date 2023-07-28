import sys
sys.path.append('..')
import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

name1 = "20220407_0831_SabrVec_test"
name2 = "test"
path1 = os.path.join('/home/fkch/mlb/data/vecdata/' + name1 + '.csv')
path2 = os.path.join('/home/fkch/mlb/' + name2 + '.csv')
reader1 = pd.read_csv(path1, sep=',')
reader2 = pd.read_csv(path2, sep=',')
df = pd.merge(reader1, reader2, on='play', how= 'left')
df_x = df.fillna({'ID': 15})
p = 0
z = 1
a_teams = ["BAL", "BOS", "CHW", "CLE", "DET", "HOU", "KCR", "LAA", "MIN", "NYY", "OAK", "SEA", "TBR", "TEX", "TOR"]
n_teams = ["ARI", "ATL", "CHC", "CIN", "COL", "LAD", "MIA", "MIL", "NYM", "PHI", "PIT", "SDP", "SFG", "STL", "WSN"]
'''
with open('/home/fkch/mlb/code/BB-RNN/MAIN/seq2vec_2000.pkl', 'rb') as f:
    params = pickle.load(f)'''

loaded_rf_model = pickle.load(open('/home/fkch/mlb/code/BB-RNN/MAIN/seq2vec_1000.pkl', 'rb'))

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
    
#df_x = df_x[df_x["Team"] == "LAA"]
#df_x["number"] = df_x['Inn.'].apply(team)
#max_num = df_x['number'].max()

#リスト初期化
m = []
tm = []
am = []
atm = []
x = []
tx = []
sequences = []
a_rmse = []
#gosa_list = []
#yerr_list = []
#avg_list = []
#gosa_np = np.empty(0)
gosa_yerr = np.empty([15, 3])
gosa_avg = np.empty(0)
np_avg = np.empty(0)
gosa_max = np.empty(0)
gosa_min = np.empty(0)
#gosas = np.empty(0)

#データ校正
for a_t in a_teams:
    df_ax = df_x[df_x["Team"] == a_t].copy()
    df_ax["number"] = df_ax['Inn.'].apply(team)
    max_num = df_ax['number'].max()
    #print(df_ax)
    for j in range(1, max_num+1):
        nm = df_ax[df_ax['number'] == j].copy()
        new_nm = df_ax[df_ax['number'] == j].copy()
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
    am.append(sequences)
    sequences = []
    atm.append(tm)
    tm = []

#test_sequences = np.array(am[0])
#scores = np.array(atm[0])
#scores = scores.astype('int64')
#print(test_sequences.shape)

    #ndarray形式にキャスト
for i, v in enumerate(a_teams):
    gosas = np.empty(0)
    gosa_np = np.empty(0)

    test_sequences = np.array(am[i])
    scores = np.array(atm[i])
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

    #予測
    #predicted = model.predict(test_sequences[0].reshape(1, 9, 13))
    y_pred = loaded_rf_model.predict(test_sequences)
    y_pred = y_pred.reshape(-1)
    #rmse = np.sqrt(mean_squared_error(scores, y_pred))
    #a_rmse.append(rmse)
    #a = loaded_rf_model.get_weights()
    #r2_score(y_test, y_pred)
    #predicted = model.predict(test_sequences)
    #結果
    for j in range(len(y_pred)):
        gosa = scores[j] - y_pred[j]
        gosa = abs(gosa)
        gosa_np = np.append(gosa_np, gosa)
    #gosa_hani = gosa_np.max() - gosa_np.min()
    g_max = gosa_np.max()
    g_min = gosa_np.min()
    #gosa_max = np.append(gosa_max, g_max)
    #gosa_min = np.append(gosa_min, g_min)
    np_avg = np.mean(gosa_np)
    gosas = np.append(gosas, [g_min, np_avg, g_max])
    #rmse = np.sqrt(mean_squared_error(scores[0], y_pred[0]))

    #print(gosas)
    gosa_yerr[i] = gosas
    #gosa_avg = np.append(gosa_avg, np_avg)
#print(gosa_yerr)
'''
fig, ax = plt.subplots()
ax.errorbar(a_teams, gosa_avg, yerr=gosa_yerr, fmt='o', ecolor='red', color='black')
ax.set_xlabel('gosa')
ax.set_ylabel('a_team')
plt.savefig('seq2vec_ateam2.png')
'''
plt.bar(range(15), gosa_yerr[:,1], width=0.6, yerr=[gosa_yerr[:,1].T-gosa_yerr[:,0].T, gosa_yerr[:,2].T-gosa_yerr[:,1].T], capsize=10, tick_label=a_teams)
plt.savefig('seq2vec_ateam_a.png')
plt.savefig('seq2vec_ateam_a.eps')