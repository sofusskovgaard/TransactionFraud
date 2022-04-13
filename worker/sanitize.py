import pandas as pd

data = pd.read_csv('./formatted.fraud.csv', index_col='step')

frauds = data.loc[data['isfraud'] == 1]
nonfraud = data.loc[data['isfraud'] == 0]

result = frauds

count = 1

while len(result.index) <= 50000:
  if count > 743:
    count = 1
  result.append(frauds.loc[count])
