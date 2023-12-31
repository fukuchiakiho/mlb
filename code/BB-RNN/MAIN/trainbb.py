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
max_counter = []

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
#print(max_num)
for jj in range(1, max_num+1):
    nm = df_x[df_x['number'] == jj]
    max_i = nm['Inn.'].max()
    for i in range(1,max_i+1):
        a = nm[nm['Inn.'] == i]
        max_counter.append(len(a))

#変数
max_sequence_len = df_x['Inn.'].max()
max_PA = max(max_counter)
#print(max_PA)
#リスト初期化
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

#入力データ

for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j]
    max_i = nm['Inn.'].max()
    re_inn = max_sequence_len - max_i
    for i in range(1,max_i+1):
        a = nm[nm['Inn.'] == i]
        re_pa = max_PA - len(a)
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
            tx = row['ID']
            tm.append(tx)
        #for b in range(re_pa):
            #tm.append(11.0)
            #m.append([])
            #mlpa = len(m)
            #for bb in range(len(m[0])):
                #m[mlpa-1].append(0.0)
    #for b in range(re_inn):
        #for bb in range(max_PA):
            #tm.append(11.0)
            #m.append([])
            #mlin = len(m)
            #for bbb in range(len(m[0])):
                #m[mlin-1].append(0.0)

#出力データ
'''
for j in range(1, max_num+1):
    nm = df_x[df_x['number'] == j]
    max_i = nm['Inn.'].max()
    re_inn = max_sequence_len - max_i
    for i in range(1,max_i+1):
        a = nm[nm['Inn.'] == i]
        for index, row in a.iterrows():
            tx = row['ID']
            tm.append(tx)
        for x in range(re_inn):
            tm.append(11.0)
            #tx = []'''

#ndarray形式にキャスト
m = np.array(m)
tm = np.array(tm)
tm = tm.astype('int64')
#m = m[30:60]
#tm = tm[:1000]

# ハイパーパラメータの設定
batch_size = 13 #
input_size = 13 #sbarの次元数
output_size = 17
hidden_size = 10  # RNNの隠れ状態ベクトルの要素数
inn_size = 13
time_size = 17  # RNNを展開するサイズ
lr = 0.1
max_epoch = 100
index_size = len(max_il)
#print(index_size)

# 学習データの読み込み

xs = m  # 入力
#print(xs.shape)
ts = tm # 出力（教師ラベル）play
#print(ts.shape)

# モデルの生成
model = SimpleRnnbb(output_size, input_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, inn_size=inn_size, time_size=time_size)
trainer.plot()

model.save_params()