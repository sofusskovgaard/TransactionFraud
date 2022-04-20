from trainer import Trainer, Predicter

trainer = Trainer(
  input='./worker/data/sanitized.fraud.csv',
  output='./worker/models'
)

trainer.run(times=5, epochs=25)

# predicter = Predicter(
#   model='./worker/models/2022-04-20_11-22-20_25'
# )

# df = pd.read_csv('./worker/data/sanitized.fraud.csv')
# # df = df[df.target == 1]
# df.pop('target')

# for row_index, row in df.iterrows():
#   input_dict = {name: tf.convert_to_tensor([value]) for name, value in row.items()}
#   prediction = predicter.predict(input_dict)
#   print(f'Prob -> {prediction:.2f}')
