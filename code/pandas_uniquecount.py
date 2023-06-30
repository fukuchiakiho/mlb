import os
import pandas as pd
name = "20220407_Angels"
path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/GAME/' + name + '.tsv')
m = -1
z = 1
s = "home"
d = "away"

def func_Inn(x):
    global m, s, d
    if x == 0:
        return s
    elif x == 1:
        return "x"
    elif x == 2:
        return "y"
    else:
        return d
    
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
    
def home(x):
    if x % 2 == 1:
        return
        
#chunksize = 1
#reader = pd.read_csv(path, chunksize=chunksize, sep='\t', usecols=['Inn.'])
#reader = pd.read_csv(path, chunksize=1, sep='\t', usecols=['play'])
reader = pd.read_csv(path, sep='\t')
#reader["ASDF"] = reader['Inn.'].apply(func_Inn)
#ff = reader['Player'].apply(func_Inn)
#ff.to_csv("try1.csv", index=False)
#dd = reader['Inn.'].apply(team)
#dd.to_csv("try12.csv", index=False)
#reader["number"] = reader['Outs'].apply(team)
#ff = reader[(reader["number"] % 2) == 0]
#ff.to_csv("try12.csv", index=False)
sep = ','
for chunk in reader["play"]:
    #t = chunk.str[:2]
    try:
        t = chunk.split(sep)
        a = t[-1:]
    except:
        a = "['NaN']"
#t = reader['play'].str[0]
#print(t)