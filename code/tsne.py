import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
import sys
sys.path.append('..')
import os
import numpy as np

name1 = "20220407_1005"
path1 = os.path.join('/home/fkch/mlb/data/fangraphs/2022/BAT/bat_20220407-20221005.csv')
reader1 = pd.read_csv(path1, sep=',')
#df = pd.merge(reader1, reader2, on='play', how= 'left')
#Freader1 = df.fillna({'ID': 15})

max_num = reader1['#'].max()
#print(max_num)

#変数
#print(max_PA)
#リスト初期化
m = []
x = []
#入力データ

for j in range(1, max_num+1):
    nm = reader1[reader1['#'] == j]
    for index, row in nm.iterrows():
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


#ndarray形式にキャスト
m = np.array(m)

#print(m.shape)

tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
X_embedded = tsne.fit_transform(m)
print(X_embedded.shape)


#ddf = pd.concat([reader1, pd.DataFrame(X_embedded, columns = ['col1', 'col2'])], axis = 1)

#article_list = ddf['Name'].unique()

#colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink"]
plt.figure(figsize = (30, 30))
for i , v in enumerate(X_embedded):
    plt.scatter(v[0],  
                v[1])

#plt.legend(fontsize = 30)
plt.savefig('tsne.png')