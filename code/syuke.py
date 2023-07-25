import os
import pandas as pd
import sys
from datetime import date, timedelta
import collections
import random
name3 = "20220407_0831_sabrvec"
m = -1
z = 1
x = []
teams = ["Angels", "Astros", "Athletics", "Blue Jays", "Braves", "Brewers", "Cardinals", "Cubs", "Diamondbacks", "Dodgers", "Giants", "Guardians", "Mariners", "Marlins", "Mets", "Nationals", "Orioles", "Padres", "Phillies", "Pirates", "Rangers", "Rays", "Red Sox", "Reds", "Rockies", "Royals", "Tigers", "Twins", "White Sox", "Yankees"]
Coulum = ['play', 'count']

def date_range(start, stop, step = timedelta(1)):
    current = start
    while current < stop:
        yield current
        current += step

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
    
def Name(x):
    l = x.split()
    ll = "".join(l[-1:])
    return ll

def Play(x):
    try:
        l = x.split(',')
        ll = "".join(l[-1:])
        return ll
    except:
        return x
    

for d in date_range(date(2022, 4, 7), date(2022, 5, 5)): #2022/04/07~2022/10/05
    tyear, tmonth, tday = str(d).split("-")
    dd = tyear+tmonth+tday
    for t in teams:
        try:
            path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/GAME/' + dd + "_" + t + '.tsv')
            reader = pd.read_csv(path, sep='\t')
            #homeチームの[play]のみ抽出
            reader["number"] = reader['Outs'].apply(team)
            reader["Player"] = reader["Player"].apply(Name)
            reader["play"] = reader["play"].apply(Play)
            ff = reader[(reader["number"] % 2) == 0]

            x += ff["play"].values.tolist()
        except:
            pass
            #print("試合なし")
#print(l)
l = collections.Counter(x)
df = pd.DataFrame(l.items(), columns=['play', 'count'])
df = df[df['count'] > 100]
df_r = df.reset_index(drop=True)
df_r['ID'] = df_r.index
df_r = df_r.fillna('Others')
#print(df_r)
df_r.to_csv('test.csv')