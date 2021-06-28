import os
import numpy as np
import tensorflow as tf
from ineuron import config as cg
import pathlib
from ineuron import config

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

def print_sample(landmark_dataset, train_ds, val_ds, test_ds, sn):
    """
    This will print out sn samples of each dataset passed
    :param landmark_dataset:
    :param test_ds:
    :param train_ds:
    :param val_ds:
    :param sn:
    """
    if landmark_dataset is not None:
        print('whole dataset sample')
        for file in landmark_dataset.take(sn).as_numpy_iterator():
            print(file)

    print('train dataset sample')
    for file in train_ds.take(sn).as_numpy_iterator():
        print(file)

    print('val dataset sample')
    for file in val_ds.take(sn).as_numpy_iterator():
        print(file)

    print('test dataset sample')
    for file in test_ds.take(sn).as_numpy_iterator():
        print(file)

    print('train_set size=', tf.data.experimental.cardinality(train_ds))
    print('val_set size=', tf.data.experimental.cardinality(val_ds))
    print('test_set size=', tf.data.experimental.cardinality(test_ds))

@tf.function
def get_ds(root_dir,train_ratio,val_ratio,file_type):
    """
    reads class files present in subfolders of root_dir and generates dataset for train, test, val
    return shape will (batchsize, tuple(ar(468,3) float32, size= 1 uint8)
    :param root_dir: folder path containing subfolders for each alertstate with .npy files in them
    :param train_ratio:
    :param val_ratio:
    :param file_type: specify it to data format either [csv or npy]
    :return: returns train, val and test datasets of type (landmark (array), label(int))
    """
    data_dir = pathlib.Path(root_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print(class_names)


    #loading list of npy files in sub directories
    landmark_dataset = tf.data.Dataset.list_files(os.path.join(root_dir, "*", "*."+file_type),shuffle=False)
    # getting count of total landmark files
    file_count = tf.data.experimental.cardinality(landmark_dataset)
    print('files count=',file_count)

    #shuffling dataset with buffersize equal to no of files and preventing reshuffle
    # while splitting train and test dataset for creating disjoint datasets
    landmark_dataset = landmark_dataset.shuffle(file_count, reshuffle_each_iteration=False)


    # (1-train_ratio+val_ratio)*dataset_size will be allocated to test_ds
    if train_ratio+val_ratio >1:
        raise Exception('set proper rations with sum <1')

    train_size =tf.cast(tf.cast(file_count,tf.float32) * train_ratio,tf.int64)
    val_size = tf.cast(tf.cast(file_count,tf.float32) * val_ratio,tf.int64)

    train_ds = landmark_dataset.take(train_size)
    test_ds = landmark_dataset.skip(train_size)
    val_ds = test_ds.take(val_size)
    test_ds = test_ds.skip(val_size)

    def get_label(file_path):
        """
        returns label extracted from path
        :param file_path:
        :return: label
        """
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == class_names
        # label = str(file_path).split("\\")[-2]
        # converting to int
        # return tf.cast(tf.strings.to_number(label), tf.uint8)
        return tf.cast(tf.argmax(one_hot), tf.uint8)

    def load_landmark_label(file_path):
        '''
        returns landmark array and label tuple
        :param file_path:
        :return: array, int
        '''
        label = get_label(file_path)  # extracting label from path basefolder
        ar = []
        if file_type == 'npy':
            ar = np.load(file_path)  # loading landmarks from numpy file
        else:
            raw = tf.io.read_file(file_path)
            lines = tf.strings.split(raw)
            dcsv = tf.io.decode_csv(lines, [0.0, 0.0, 0.0])
            ar = tf.stack([dcsv[0] / 1000, dcsv[1] / 1000, dcsv[2] / 1000], axis=1)

        try:
            ar = tf.reshape(ar, [468, 3])
        except Exception as e:
            print(e," file path=",file_path)


        # label = tf.reshape(label,[1])
        return ar, label

    # print_sample(landmark_dataset, train_ds, val_ds, test_ds, 0)
    #mapping file_path in dataset to landmark (array), label (int)
    # train_ds = train_ds.map(lambda item: tf.numpy_function(
    #           load_landmark_label, [item], [tf.float32, tf.uint8]),
    #           num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS)
    # val_ds = val_ds.map(lambda item: tf.numpy_function(
    #           load_landmark_label, [item], [tf.float32, tf.uint8]),
    #           num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS)
    # test_ds = test_ds.map(lambda item: tf.numpy_function(
    #           load_landmark_label, [item], [tf.float32, tf.uint8]),
    #           num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS)

    train_ds = train_ds.map(load_landmark_label,
                            num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS,drop_remainder=True)
    val_ds = val_ds.map(load_landmark_label,
                        num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS,drop_remainder=True)
    test_ds = test_ds.map(load_landmark_label,
                          num_parallel_calls=tf.data.AUTOTUNE).batch(config.BS,drop_remainder=True)

    print("printing sample from full size datasets")
    # print_sample(landmark_dataset, train_ds, val_ds, test_ds, 1)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # root_dir = r'..\data'
    root_dir = None
    get_ds(root_dir,0.7,0.1,'csv')
