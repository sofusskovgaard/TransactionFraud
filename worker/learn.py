import time
from oauthlib import get_debug

import pandas as pd
import numpy as np
import tensorflow as tf

def get_data(file_path,target='isfraud'):
  d = pd.read_csv(file_path)
  d['target'] = np.where(d[target]==1, 1, 0)
  d = d.drop(columns=[target])
  return d

def split_data(data):
  train, val, test = np.split(data.sample(frac=1), [int(0.8*len(data)), int(0.9*len(data))])
  print(len(train), 'training examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')
  return {'train':train,'val':val,'test':test}

def df_to_dataset(dataframe, shuffle=True, batch_size=32, should_batch=True):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  if should_batch:
    ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

def create_datasets(sdata):
  train = df_to_dataset(sdata['train'], should_batch=False)
  val = df_to_dataset(sdata['val'], shuffle=False, should_batch=False)
  test = df_to_dataset(sdata['test'], shuffle=False, should_batch=False)
  return {'train':train,'val':val,'test':test}

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = tf.keras.layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
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

def preprocess_data(training_dataset):
  all_inputs = []
  encoded_features = []

  # Numerical features.
  for header in ['step', 'hourofday', 'amount', 'oldbaldest', 'newbaldest', 'errbaldest', 'oldbalorg', 'newbalorg', 'errdbalorg']:
    print(f'processing {header} feature')
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, training_dataset)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

  print('processing type feature')
  type_col = tf.keras.Input(shape=(1,), name='type', dtype='string')

  encoding_layer = get_category_encoding_layer(name='type',
                                              dataset=training_dataset,
                                              dtype='string',
                                              max_tokens=5)
  encoded_type_col = encoding_layer(type_col)
  all_inputs.append(type_col)
  encoded_features.append(encoded_type_col)
  return [all_inputs, encoded_features]

def create_model(all_inputs, encoded_features):
  all_features = tf.keras.layers.concatenate(encoded_features)
  x = tf.keras.layers.Dense(128, activation="relu")(all_features)
  x = tf.keras.layers.Dense(128, activation="relu")(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  output = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(all_inputs, output)

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["accuracy"])
  return model

print('reading data')
raw_data = get_data('./formatted.fraud.mini.csv')
print('splitting data')
data = split_data(raw_data)
print('creating datasets')
datasets = create_datasets(data)

print('preprocessing data')
all_inputs, encoded_features = preprocess_data(datasets['train'])

print('creating model')
model = create_model(all_inputs, encoded_features)

model.fit(datasets['train'], epochs=10, validation_data=datasets['val'])

loss, accuracy = model.evaluate(datasets['test'])
print("Accuracy", accuracy)
print("Loss", loss)

ts = time.strftime("%Y_%m_%d-%H_%M_%S")
model.save(f'models/{ts}')
