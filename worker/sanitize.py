from tokenize import String
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import path

class Sanitizer:
  def __init__(self, input, output):
    self.input = input
    self.output = output

  def add_hour_of_day(self, df: pd.DataFrame):
    arr = []
    for row in df.iterrows():
      arr.append(row[1].step % 24)
    df.insert(1, "hourOfDay", arr)

  def add_err_balances(self, df: pd.DataFrame):
    errbalanceOrig = []
    errbalanceDest = []
    for row in df.iterrows():
      errbalanceOrig.append(row[1].newbalanceOrig + row[1].amount - row[1].oldbalanceOrg)
      errbalanceDest.append(row[1].newbalanceDest + row[1].amount - row[1].oldbalanceDest)
    df.insert(len(df.columns)-1, "errbalanceOrig", errbalanceOrig)
    df.insert(len(df.columns)-1, "errbalanceDest", errbalanceDest)

  def remove_step(self, df: pd.DataFrame):
    df = df.drop(axis=1,columns=['step'])
    return df

  def remove_unneeded_columns(self, df: pd.DataFrame):
    df = df.drop(axis=1,columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    df = df.rename(columns={'isFraud':'target'})
    return df

  def remove_unneeded_rows(self, df: pd.DataFrame):
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

  def run(self):
    if path.exists(self.input):
      data = pd.read_csv(self.input)
    else:
      data = pd.read_csv(self.output)
      data = self.remove_unneeded_columns(data)
      data = self.remove_unneeded_rows(data)
      self.add_hour_of_day(data)
      data = self.remove_step(data)
      self.add_err_balances(data)
      data.to_csv(self.input, index=False)
    return data

def visualize(df: pd.DataFrame):
  sns.countplot(x="type", data=df, hue="target", palette="Set2")
  plt.show()

sanitizer = Sanitizer(
  input='./worker/data/sanitized.fraud.csv',
  output='./worker/data/fraud.csv'
)

data = sanitizer.run()
visualize(data)