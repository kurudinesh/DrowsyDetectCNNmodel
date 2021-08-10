import tensorflow as tf
path = "saved_data"

# Save a dataset
dataset = tf.data.Dataset.range(2)
tf.data.experimental.save(dataset, path)
new_dataset = tf.data.experimental.load(path)
for elem in new_dataset:
  print(elem)


#flat_map code error below
filenames = [path]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
def parse_fn(filename):
  print(filename)
  return tf.data.experimental.load(filename)
dataset = dataset.flat_map(lambda x:
    parse_fn(x))

for item in dataset.as_numpy_iterator():
    print(item)

#interleave code error below
filenames = [path]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
def parse_fn(filename):
  print(filename)
  return tf.data.experimental.load(filename)
dataset = dataset.interleave(
    parse_fn)

for item in dataset.as_numpy_iterator():
    print(item)