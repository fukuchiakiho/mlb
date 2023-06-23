import os
import pandas as pd
name = "20220407_Angels"
path = os.path.join('/home/fkch/mlb/data/fangraphs/2022/' + name + '.tsv')

chunksize = 1
reader = pd.read_table(path, chunksize=chunksize, sep='\t')
for chunk in reader:
    print(chunk.iloc[1,1])
#a