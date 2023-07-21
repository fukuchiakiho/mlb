# coding: utf-8
import sys
sys.path.append('..')
import os
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
import pandas as pd
from simple_rnnbb import SimpleRnnbb
import numpy as np

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
max_mask = []

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
    
def masking(x, max_sequence_len):
    for leng in x:
        for lleng in leng:
            size = max_sequence_len - len(lleng)
            for i in range(size):
                lleng.append(0)



df_x["number"] = df_x['Inn.'].apply(team)
max_num = df_x['number'].max()
df_x.to_csv("try.csv", index=False)

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
max_counter = []

#入力データ
for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j]
    max_i = nm['Inn.'].max()
    for i in range(1,max_i+1):
        a = nm[nm['Inn.'] == i]
        for index, row in a.iterrows():
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
            #tx = []
        l.append(m)
        m = []
    all_l.append(l)
    l = []

for i in all_l:
    #max_counter.append(len(i))
    for ii in i:
        max_counter.append(len(ii))
max_sequence_len = max(max_counter)#各イニングの打席数の最大値
#print(max_sequence_len)

#出力データ
'''
for index, row in df_x.iterrows():
    row['ID'] = int(row['ID'])
    t.append(row['ID'])
'''
for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j]
    max_i = nm['Inn.'].max()
    for i in range(1,max_i+1):
        a = nm[nm['Inn.'] == i]
        for index, row in a.iterrows():
            tx = row['ID']
            tm.append(tx)
            #tx = []
        tl.append(tm)
        tm = []
    all_tl.append(tl)
    tl = []
    max_il.append(max_i)


# ハイパーパラメータの設定
batch_size = 10
input_size = 13 #sbarの次元数
output_size = 112555
hidden_size = 10  # RNNの隠れ状態ベクトルの要素数
#time_size = 5  # RNNを展開するサイズ
lr = 0.1
max_epoch = 20
index_size = len(max_il)

# 学習データの読み込み
#all_l = masking(all_l, max_sequence_len)
for i in range(index_size):
    x = max_sequence_len - len(all_l[i])
    for j in range(x):
        all_l[i].append([])
        all_tl[i].append([])
'''
for inn in range(len(batch_x)):
                    i = max_inns - len(batch_x[inn])
                    for x in range(i):
                        batch_x[inn].append([0.0]*13)
                        batch_t[inn].append(11.0)'''

xs = all_l  # 入力
#xs = np.array(xs, dtype=object)
#print(xs[148])
ts = all_tl # 出力（教師ラベル）play
#ts = np.array(ts,  dtype=object)
#print(ts[0])

# モデルの生成
model = SimpleRnnbb(output_size, input_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, max_iter=index_size, max_inns=max_sequence_len)
trainer.plot()

model.save_params()