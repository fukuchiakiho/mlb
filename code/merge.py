import pandas as pd

reader1 = pd.read_csv("try12.csv", sep=',')
reader2 = pd.read_csv("try22.csv", sep=',')

a = reader1.merge(reader2, on="Player", how="left")
a.to_csv("try123.csv", index=False)