import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml
import os

class Trainer:
  def __init__(self, input, output):
    self.input = input
    self.output = output
    self.raw_data = pd.read_csv(input)

  def __split(self):
    train, val, test = np.split(self.raw_data.sample(frac=1), [int(0.8*len(self.raw_data)), int(0.9*len(self.raw_data))])
    self.split_data = {'train':train,'val':val,'test':test}

  def __create_dataset(self, df: pd.DataFrame, batch_size=128):
    df = df.copy()
    labels = df.pop('target')
    df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

  def __create_datasets(self):
    train = self.__create_dataset(self.split_data['train'])
    val = self.__create_dataset(self.split_data['val'])
    test = self.__create_dataset(self.split_data['test'])
    self.datasets = {'train':train,'val':val,'test':test}

  def __create_numeric_layer(self, name, dataset, configs: dict):
    # Create a Normalization layer for the feature.
    normalizer = keras.layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    configs.update({ name: {'mean':float(normalizer.mean[0][0]),'variance':float(normalizer.variance[0][0])} })

    return normalizer

  def __create_string_layer(self, name, dataset, dtype, configs: dict, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
      index = keras.layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
      index = keras.layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=4)

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    configs.update({ name: {'vocab':list(map(lambda x: str(x), index.get_vocabulary()))} })

    # Encode the integer indices.
    encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

  def __preprocess_data(self):
    all_inputs = []
    encoded_features = []

    numerical_features = ['hourOfDay', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errbalanceOrig', 'errbalanceDest']
    string_features = ['type']

    configs = {}

    # Numerical features.
    for header in numerical_features:
      print(f'Processing column {header}')
      col = keras.Input(shape=(1,), name=header)
      layer = self.__create_numeric_layer(header, self.datasets['train'], configs)
      all_inputs.append(col)
      encoded_features.append(layer(col))

    # String features
    for header in string_features:
      print(f'Processing column {header}')
      col = keras.Input(shape=(1,), name=header, dtype='string')
      layer = self.__create_string_layer(header, self.datasets['train'], 'string', configs)
      all_inputs.append(col)
      encoded_features.append(layer(col))

    self.configs = configs
    self.all_inputs = all_inputs
    self.encoded_features = encoded_features

  def __create_model(self):
    all_features = keras.layers.concatenate(self.encoded_features)
    x = keras.layers.Dense(9, activation="relu")(all_features)
    x = keras.layers.Dense(9, activation="relu")(x)
    output = keras.layers.Dense(1, activation="relu")(x)

    model = keras.Model(self.all_inputs, output)

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    self.model = model

  def __fit_model(self, epochs):
    self.model.fit(self.datasets['train'], epochs=epochs, validation_data=self.datasets['val'])

  def __save_model(self, ts, suffix):
    self.model.save(f'{self.output}/{ts}{suffix}')

  def __save_model_metadata(self, ts, accuracy, loss, epochs):
    with open(f'{self.output}/{ts}_{epochs}/metadata.yaml', 'w') as stream:
      yaml.dump({
        'accuracy':float(accuracy),
        'loss': float(loss),
        'epochs': epochs,
        'configs': self.configs
      }, stream, sort_keys=False)

  def run(self, times=1, epochs=10, save_each=True):
    self.__split()
    self.__create_datasets()
    self.__preprocess_data()
    self.__create_model()
    count = 0

    while count < times:
      os.system('clear')
      self.__fit_model(epochs)
      count+=1
      if save_each:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.__save_model(ts, f'_{count*epochs}')
        loss, accuracy = self.model.evaluate(self.datasets['test'], verbose=0)
        self.__save_model_metadata(ts, accuracy, loss, count*epochs)
      elif count+1 == times:
        ts = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.__save_model(ts, f'_{count*epochs}')
        loss, accuracy = self.model.evaluate(self.datasets['test'])
        self.__save_model_metadata(ts, accuracy, loss, count*epochs)

class Predicter:
  def __init__(self, model):
    self.model = tf.keras.models.load_model(model)
  
  def predict(self, data):
    pred = self.model.predict(data)
    return tf.nn.sigmoid(pred[0])[0] * 100
  