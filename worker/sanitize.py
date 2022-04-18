import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import path

pd.set_option('display.float_format', lambda x: '%.5f' % x)

def add_hour_of_day(df: pd.DataFrame):
  arr = []
  for row in df.iterrows():
    arr.append(row[1].step % 24)
  df.insert(1, "hourOfDay", arr)

def add_err_balances(df: pd.DataFrame):
  errbalanceOrig = []
  errbalanceDest = []
  for row in df.iterrows():
    errbalanceOrig.append(row[1].newbalanceOrig + row[1].amount - row[1].oldbalanceOrg)
    errbalanceDest.append(row[1].newbalanceDest + row[1].amount - row[1].oldbalanceDest)
  df.insert(len(df.columns)-1, "errbalanceOrig", errbalanceOrig)
  df.insert(len(df.columns)-1, "errbalanceDest", errbalanceDest)

def remove_step(df: pd.DataFrame):
  df = df.drop(axis=1,columns=['step'])
  return df

def remove_unneeded_columns(df: pd.DataFrame):
  df = df.drop(axis=1,columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
  df = df.rename(columns={'isFraud':'target'})
  return df

def remove_unneeded_rows(df: pd.DataFrame):
  df = pd.concat([df.loc[df['type'] == "CASH_OUT"], df.loc[df['type'] == "TRANSFER"]])
  result = pd.DataFrame(df.loc[df['target'] == 1])
  chunks = df.loc[df['target'] == 0].groupby(['step', 'type'])

  count = 0

  while len(result) <= 50000:
    for chunk in chunks:
      try:
        result = pd.concat([result, chunk[1].iloc[[count]]])
        df = df.iloc[count+1:]
      except Exception:
        pass
    count += 1
  return result.sort_values('step', ignore_index=True)

def sanitize():
  if path.exists('./worker/data/formatted.fraud.csv'):
    data = pd.read_csv('./worker/data/formatted.fraud.csv')
  else:
    data = pd.read_csv('./worker/data/fraud.csv')
    data = remove_unneeded_columns(data)
    # print(data.head())
    data = remove_unneeded_rows(data)
    add_hour_of_day(data)
    data = remove_step(data)
    add_err_balances(data)
    data.to_csv('./worker/data/formatted.fraud.csv', index=False)
  return data

def visualize(df: pd.DataFrame):
  sns.countplot(x="type", data=df, hue="target", palette="Set2")
  plt.show()

def normalize(df: pd.DataFrame):
  # col = df['amount'].to_numpy()
  # print(col)
  # norm = np.linalg.norm(col)
  # print(norm)
  # print(col/norm)

  for col in df.columns:
    if (df.dtypes[col] == "int64" or df.dtypes[col] == "float64"):
      data = df[col].to_numpy()
      # print(data)
      norm = np.linalg.norm(data)
      print(f"{col} = {norm}")
      df[col] = data/norm
      print(df[col])
  df.to_csv('./worker/data/normalized.formatted.fraud.csv', index=False, float_format='%.8f')
  return df

def normalize_type(df: pd.DataFrame):
  from tensorflow import keras
  data = df["type"]

  index = keras.layers.StringLookup(max_tokens=None)
  index.adapt(data)
  encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  arr = []
  for item in data:
    arr.append(encoder(index(item)))
  df["type"] = arr
  return df

data = sanitize()
# visualize(data)

# print(data.dtypes)

normalized_data = normalize(data)
normalized_data = normalize_type(normalized_data)
print(normalized_data)