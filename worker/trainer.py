import time
import pandas as pd
import numpy as np
import tensorflow as tf

class Trainer:
  def __init__(self, path):
    self.path = path

  def __create_dataset(self, dataframe: pd.DataFrame, batch_size=128):
    df = dataframe.copy()
    print(df)
    labels = df.pop('target')
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

  def __normalization_layer(self, name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = tf.keras.layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

  def __category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
      index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
      index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=4)

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

  def load_data(self):
    self.raw_data = pd.read_csv(self.path)

  def create_datasets(self):
    train, val, test = np.split(self.raw_data.sample(frac=1), [int(0.8*len(self.raw_data)), int(0.9*len(self.raw_data))])
    # datasets = [self.__create_dataset(item) for item in splat]

    self.datasets = {
      'train': self.__create_dataset(train),
      'val': self.__create_dataset(val),
      'test': self.__create_dataset(test)
    }

  def preprocess_data(self):
    numerical_features = []
    string_features = []

    for col in self.raw_data.columns:
      if (self.raw_data.dtypes[col] == "int64" or self.raw_data.dtypes[col] == "float64"):
        numerical_features.append(col)
      elif (self.raw_data.dtypes[col] == "string"):
        string_features.append(col)

    self.all_inputs = []
    self.encoded_features = []

    # Numerical features.
    for i,num_header in enumerate(numerical_features):
      print(f'processing feature {num_header} ({i+1}/{len(numerical_features)})')
      col = tf.keras.Input(shape=(1,), name=num_header)
      num_layer = self.__normalization_layer(num_header, self.datasets['train'])
      num_encoded_col = num_layer(col)
      self.all_inputs.append(col)
      self.encoded_features.append(num_encoded_col)

    for i,str_header in enumerate(string_features):
      print(f'processing feature {str_header} ({i+1}/{len(string_features)})')
      col = tf.keras.Input(shape=(1,), name=str_header, dtype='string')
      str_layer = self.__category_encoding_layer(name=str_header, dataset=self.datasets['train'], dtype='string', max_tokens=5)
      str_encoded_col = str_layer(col)
      self.all_inputs.append(col)
      self.encoded_features.append(str_encoded_col)

  def create_model(self):
    x = tf.keras.layers.concatenate(self.encoded_features)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.15)(x)
    # x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(self.all_inputs, output)

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    return model

  def save_model(self, name, model):
    model.save(f'models/{name}')
  
  def save_model_metadata(self, ts, accuracy, loss, epochs):
    with open(f'models/{ts}-{epochs}/readme.txt', 'w') as f:
        f.write(f'accuracy\t{accuracy}\n')
        f.write(f'loss\t\t\t{loss}\n')
        f.write(f'epochs\t\t{epochs}\n')

  def load_model(self, name):
    return tf.keras.models.load_model(f'models/{name}')

  def run(self):
    self.load_data()
    self.create_datasets()
    self.preprocess_data()
    model = self.create_model()

    epochs = 0
    epoch_batch = 10

    while epochs < 100:
      model.fit(self.datasets['train'], epochs=epoch_batch, validation_data=self.datasets['val'])
      epochs += epoch_batch
      metrics = model.evaluate(self.datasets['test'])

      ts = time.strftime("%Y_%m_%d-%H_%M_%S")
      self.save_model(f'{ts}-{epochs}', model)
      self.save_model_metadata(ts, metrics[1], metrics[0], epochs)
