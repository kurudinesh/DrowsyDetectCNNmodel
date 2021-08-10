import tensorflow as tf
path = "saved_data_2"

# Save a dataset
dataset = tf.data.Dataset.range(10)
tf.data.experimental.save(dataset, path)
new_dataset = tf.data.experimental.load(path)
for elem in new_dataset:
  print(elem)


#numpy iterator code throws error inside numpy_function
filenames = [path]
dataset = tf.data.Dataset.from_tensor_slices(filenames)


def parse_fn(filename):
    part_ds = tf.data.experimental.load(bytes.decode(filename),compression='GZIP')
    ar = []
    for i in part_ds.as_numpy_iterator():
        ar.append(i)
    return ar

dataset = dataset.interleave(lambda x:tf.data.Dataset.from_tensor_slices(
            tf.numpy_function(parse_fn,[x],[tf.int64])),
            cycle_length=4, block_length=16)

for item in dataset.as_numpy_iterator():
    print(item)