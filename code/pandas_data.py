import os
import pandas as pd
name = "20220407_Angels"
path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/GAME/' + name + '.tsv')
m = -1
z = 1
s = "home"
d = "away"
a = pd.DataFrame(index=[])

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
        
def Name(x):
    l = x.split()
    ll = "".join(l[-1:])
    return ll

#chunksize = 1
#reader = pd.read_csv(path, chunksize=chunksize, sep='\t', usecols=['Inn.'])
#reader = pd.read_csv(path, chunksize=1, sep='\t', usecols=['Inn.'])
reader = pd.read_csv(path, sep='\t')
#reader["ASDF"] = reader['Inn.'].apply(func_Inn)
#ff = reader['Player'].apply(func_Inn)
#ff.to_csv("try1.csv", index=False)
#dd = reader['Inn.'].apply(team)
#dd.to_csv("try12.csv", index=False)
reader["number"] = reader['Outs'].apply(team)
reader["Player"] = reader["Player"].apply(Name)
ff = reader[(reader["number"] % 2) == 0]
a = ff[["Inn.", "Player"]]
a.to_csv("try12.csv", index=False)
#for chunk in reader:
#    if chunk['Inn'] == '1':
#        print(chunk)
#print(reader)