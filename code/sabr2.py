import os
import pandas as pd
import sys
from datetime import date, timedelta
import collections
import random
df_dateL = pd.DataFrame(columns=['learn'], dtype=object)
df_dateT = pd.DataFrame(columns=['test'], dtype=object)
m = -1
z = 1
x = []
learn = []
test = []
teams = ["Angels", "Astros", "Athletics", "Blue Jays", "Braves", "Brewers", "Cardinals", "Cubs", "Diamondbacks", "Dodgers", "Giants", "Guardians", "Mariners", "Marlins", "Mets", "Nationals", "Orioles", "Padres", "Phillies", "Pirates", "Rangers", "Rays", "Red Sox", "Reds", "Rockies", "Royals", "Tigers", "Twins", "White Sox", "Yankees"]
cols = ['Inn.', 'Player', 'Score', 'play', 'Team', 'G', 'PA', 'HR', 'R', 'RBI', 'SB', 'BB%', 'K%', 'ISO', 'BABIP', 'AVG', 'OBP', 'SLG', 'wOBA', 'xwOBA', 'wRC+', 'BsR', 'Off', 'Def', 'WAR', 'Unnamed: 23']
name = "20220407_0831_SabrVec_learn"
outpath = os.path.join('/home/fkch/mlb/data/vecdata/' + name + '.csv')
name2 = "20220407_0831_SabrVec_test"
outpath2 = os.path.join('/home/fkch/mlb/data/vecdata/' + name2 + '.csv')
datepath1 = os.path.join('/home/fkch/mlb/data/vecdata/learn_date.csv')
datepath2 = os.path.join('/home/fkch/mlb/data/vecdata/test_date.csv')
df = pd.DataFrame(columns=cols, dtype=object)
df2 = pd.DataFrame(columns=cols, dtype=object)

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
    N = l[0][0] #David
    M = "".join(l[-1:])
    return N + " " + M

def Play(x):
    try:
        l = x.split(',')
        ll = "".join(l[-1:])
        return ll
    except:
        return x
    
def Score(x):
    s = x[0]
    return s
    

for d in date_range(date(2022, 5, 1), date(2022, 10, 5)): #2022/04/07~2022/10/05
    tyear, tmonth, tday = str(d).split("-")
    dd = tyear+tmonth+tday
    r = random.random()
    if r < 0.7:
        learn.append(tmonth+tday)
        for t in teams:
            try:
                path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/GAME/' + dd + "_" + t + '.tsv')
                path1 = os.path.join('/home/fkch/mlb/data/fangraphs/2022/BAT/bat_20220407-' + dd + '.csv')
                game = pd.read_csv(path, sep='\t')
                bat = pd.read_csv(path1, sep=',')
                #homeチームの[play]のみ抽出
                game["number"] = game['Outs'].apply(team)
                game["Player"] = game["Player"].apply(Name)
                game["play"] = game["play"].apply(Play)
                game["Score"] = game["Score"].apply(Score)
                ff = game[(game["number"] % 2) == 0]
                new_game = ff[["Inn.", "Player", "Score", "play"]]
                #print(new_game)
                #セイバー指標抽出
                bat["Name"] = bat["Name"].apply(Name)
                new_bat = bat.rename({'Name': 'Player'}, axis=1)
                new_bat = new_bat.drop('#', axis=1)
                sbar = new_game.merge(new_bat, on='Player', how='left')
                #print(sbar)
                df = pd.concat([df, sbar], axis=0)
            except:
                pass
    else:
        test.append(tmonth+tday)
        for t in teams:
            try:
                path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/GAME/' + dd + "_" + t + '.tsv')
                path1 = os.path.join('/home/fkch/mlb/data/fangraphs/2022/BAT/bat_20220407-' + dd + '.csv')
                game = pd.read_csv(path, sep='\t')
                bat = pd.read_csv(path1, sep=',')
                #homeチームの[play]のみ抽出
                game["number"] = game['Outs'].apply(team)
                game["Player"] = game["Player"].apply(Name)
                game["play"] = game["play"].apply(Play)
                game["Score"] = game["Score"].apply(Score)
                ff = game[(game["number"] % 2) == 0]
                new_game = ff[["Inn.", "Player", "Score", "play"]]
                #print(new_game)
                #セイバー指標抽出
                bat["Name"] = bat["Name"].apply(Name)
                new_bat = bat.rename({'Name': 'Player'}, axis=1)
                new_bat = new_bat.drop('#', axis=1)
                sbar = new_game.merge(new_bat, on='Player', how='left')
                df2 = pd.concat([df2, sbar], axis=0)
            except:
                pass
            #print("試合なし")
df.to_csv(outpath)
print(df)
df2.to_csv(outpath2)
df_dateL['learn'] = learn
df_dateT['test'] = test
df_dateL.to_csv(datepath1)
df_dateT.to_csv(datepath2)
'''
l = collections.Counter(x)
df = pd.DataFrame(l.items(), columns=['play', 'count'])
df = df[df['count'] > 100]
df_r = df.reset_index(drop=True)
df_r['ID'] = df_r.index
df_r = df_r.fillna('Others')
#print(df_r)
df_r.to_csv('test.csv')
'''