# import the necessary packages
from ineuron import config
from ineuron import utils
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import tensorflow as tf
import os
import pathlib
import numpy as np
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
tf.data.Dataset.concatenate()
#set it to true for debugging datset loader
tf.config.run_functions_eagerly(False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tuner", required=False, type=str,
                choices=["hyperband", "random", "bayesian"],
                default='random',
                help="type of hyperparameter tuner we'll be using, DEFAULT=random")
ap.add_argument("-d", "--data", required=True,
                help="path to data directory")
ap.add_argument("-e", "--env", required=False,
                choices=["local", "cloud"],
                default='local',
                help="if local is used then training happens on Batch_size*localcount records, DEFAULT local",
                )
ap.add_argument("-o", "--output", required=False,
                default='output',
                help="saves tensorboard logs, checkpoints, hyperparameter logs and trained model in this folder, DEFAULT output",
                )
ap.add_argument("-lc", "--localcount", required=False, type=int,
                default=1000,
                help="in local mode the length of ds =batchsize*localcount,\n"
                        +" if count is -1 then whole data will be loaded, Default 1000",
                )
ap.add_argument("-f", "--format", required=False, type=str,
                choices=["csv", "npy",'dataframe'],
                default='dataframe',
                help="file format landmarks of video frames are saved in data directory, DEFAULT 'dataframe'"
                )
ap.add_argument("-m", "--mode", required=False, type=str,
                choices=['tune','train','test'],
                default='train',
                help="enter 'tune' for hyperparameter tuning or others, default train"
                )
ap.add_argument("-fld", "--fold", required=False, type=int,
                choices=[1,2,3,4,5],
                default=5,
                help="Enter fold to use for testing and remaining 4 folds will be used for training,Default: 5"
                )
ap.add_argument("-vp", "--valpart", required=False, type=int,
                choices=[1,2],
                default=2,
                help="Enter validationpart to select as validation"
                )

args = vars(ap.parse_args())

#setting output_base_folder_path
output_path = args['output']
tuning_mode = args["tuner"]
localcount = args['localcount']
file_type = args['format']
env = args['env']
root_dir = args['data']
mode = args['mode']
fold = args['fold']
val_part = args['valpart']
#setting path for saving plot of training loss, accuracy
plot_path = os.path.join(output_path,tuning_mode+'_plot.png')


#setting logs, checkpoints, data drirectories
tensorboard_log_dir = os.path.join(output_path, 'logs')
model_checkpoint_dir = os.path.join(output_path,"checkpoints")
model_path = os.path.join(output_path,config.modelname)

#getting classes
data_dir = pathlib.Path(root_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

for item in data_dir.glob(os.path.join('*','*')):
    print("datasets are", item)

# @tf.function
def get_ds():
    """
    reads class files present in subfolders of root_dir and generates dataset for train dataset
    return shape will (batchsize, tuple(ar(468,3) float32, size= 1 uint8)
    :return: returns train dataset of type (landmark (array), label(int))
    """

    #loading list of npy files in sub directories
    landmark_dataset = tf.data.Dataset.list_files(os.path.join(root_dir, "*", "*."+file_type),shuffle=False)
    # getting count of total landmark files
    file_count = tf.data.experimental.cardinality(landmark_dataset)
    tf.print('files count=',file_count)

    #shuffling dataset with buffersize equal to no of files and preventing reshuffle
    # while splitting train and test dataset for creating disjoint datasets
    landmark_dataset = landmark_dataset.shuffle(file_count, reshuffle_each_iteration=False)


    train_ds = landmark_dataset

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
        shape = [468, 3]
        ar = tf.zeros(
                    shape, dtype=tf.dtypes.float32, name=None
                    )
        label = tf.cast(tf.uint8.max, tf.uint8)
        try:
            label = get_label(file_path)  # extracting label from path basefolder

            if file_type == 'npy':
                ar = np.load(file_path)  # loading landmarks from numpy file
            else:
                raw = tf.io.read_file(file_path)
                lines = tf.strings.split(raw)
                # if lines.shape[0]==468:
                dcsv = tf.io.decode_csv(lines, [0.0, 0.0, 0.0])
                ar = tf.stack([dcsv[0] / 1000, dcsv[1] / 1000, dcsv[2] / 1000], axis=1)
                ar = tf.reshape(ar, [468, 3])
        except:
            print("Error in file path=",file_path)
            # raise e

        # label = tf.reshape(label,[1])
        return ar, label

    train_ds = train_ds.map(load_landmark_label,
                            num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

    return train_ds, file_count

def get_ds_from_dataset(folds,part=None,take_count=None):
    """
    reads class files present in subfolders of root_dir and generates dataset for train dataset
    return shape will (batchsize, tuple(ar(468,3) float32, size= 1 uint8)
    :return: returns train dataset of type (landmark (array), label(int))
    """
    path_lists = []
    if part is not None:
        fold = folds[0]
        offset = (part-1)*6
        start = (fold - 1) * 12 + offset+1
        end = start + 6
        for subject in range(start, end):
            path_lists.append(os.path.join(root_dir, "*", '{:02d}'.format(subject)))
    else:
        for f in folds:
            start = (f-1)*12+1
            end = f*12+1
            for subject in range(start,end):
                path_lists.append(os.path.join(root_dir, "*", '{:02d}'.format(subject)))

    print('searching for datasets at path',path_lists)

    #loading list of npy files in sub directories
    landmark_dataset = tf.data.Dataset.list_files(path_lists,shuffle=False)
    # getting count of total landmark files
    file_count = tf.data.experimental.cardinality(landmark_dataset)
    tf.print('dataset files count=',file_count)

    #shuffling dataset with buffersize equal to no of files and preventing reshuffle
    # while splitting train and test dataset for creating disjoint datasets
    landmark_dataset = landmark_dataset.shuffle(file_count, reshuffle_each_iteration=False)

    def get_label(file_path):
        """
        returns label extracted from path
        :param file_path:
        :return: label
        """
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == class_names
        return tf.cast(tf.argmax(one_hot), tf.uint8)

    train_ds = None

    for i in landmark_dataset.as_numpy_iterator():
        print('loading dataset',i)
        label = get_label(i)
        print(label)
        try:
            part_ds = tf.data.experimental.load(bytes.decode(i), compression='GZIP')
        except:
            continue
        def maplabel(x):
            return x,label
        part_ds = part_ds.map(maplabel)
        if train_ds ==None:
            train_ds = part_ds
        else:
            train_ds = train_ds.concatenate(part_ds)

    # train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

    ds_count = tf.data.experimental.cardinality(train_ds)

    print("ds count",ds_count)

    train_ds = train_ds.shuffle(ds_count, reshuffle_each_iteration=False)

    if take_count is not None:
        train_ds = train_ds.batch(config.BS).take(take_count).cache()
    else:
        train_ds = train_ds.batch(config.BS).cache()
    print("batched ds size=", tf.data.experimental.cardinality(train_ds))

    return train_ds, file_count

def build_model(hp):
    '''
    used for returning a hyperpapameter DD2CNN model
    :param hp: hyperparameter object
    :return: hypermodel
    '''
    # initialize the model along with the input shape
    model = Sequential()
    inputShape = config.INPUT_SHAPE

    # first CONV1 => leakyRELU => MAXPOOL1 => dropout layer set
    filter_1 = hp.Int("filter_1", min_value=32, max_value=128, step=32)
    kernel_1 = hp.Int("kernel_1", min_value=3, max_value=9, step=2)
    leaky_alpha_1 = hp.Float('leaky_alpha_1', min_value=0, max_value=0.5, step=0.1)
    # pool_1 = hp.Int("pool_1", min_value=2, max_value=3, step=2)
    pool_1 = 2
    dropout_1 = hp.Float('dropout_1', min_value=0.05, max_value=0.5, step=0.1)

    model.add(Conv1D(filter_1, kernel_1, padding="same", input_shape=inputShape))
    model.add(LeakyReLU(alpha=leaky_alpha_1))
    model.add(MaxPooling1D(pool_size=pool_1))
    model.add(Dropout(dropout_1))

    #layer 2-4 CONV1 => leakyRELU => MAXPOOL1 => dropout layer set

    filter_x = hp.Int("filter_x", min_value=128, max_value=1048, step=256)
    kernel_x = hp.Int("kernel_x", min_value=3, max_value=9, step=2)
    leaky_alpha_x = hp.Float('leaky_alpha_x', min_value=0, max_value=0.5, step=0.1)
    # pool_x = hp.Int("pool_x", min_value=2, max_value=3, step=2)
    pool_x =2
    dropout_x = hp.Float('dropout_x', min_value=0.05, max_value=0.5, step=0.1)

    for _ in range(3):
        model.add(Conv1D(filter_x, kernel_x, padding="same"))
        model.add(LeakyReLU(alpha=leaky_alpha_x))
        model.add(MaxPooling1D(pool_size=pool_x,strides=2))
        model.add(Dropout(dropout_x))

    # classifier flatten => FC => softmax layers
    model.add(Flatten())
    model.add(Dense(config.NUM_CLASSES,activation="softmax"))

    # initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-2, 1e-3])
    opt = Adam(learning_rate=lr)
    metric =SparseCategoricalAccuracy()
    # compile the model
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                  metrics=metric)
    # return the model
    return model

def hp_tune():
    # check if we will be using the hyperband tuner
    if tuning_mode == "hyperband":
        # instantiate the hyperband tuner object
        print("[INFO] instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            build_model,
            objective=config.objective,
            max_epochs=config.EPOCHS,
            factor=3,
            seed=42,
            directory=output_path,
            project_name=tuning_mode)
    # check if we will be using the random search tuner
    elif tuning_mode == "random":
        # instantiate the random search tuner object
        print("[INFO] instantiating a random search tuner object...")
        tuner = kt.RandomSearch(
            build_model,
            objective=config.objective,
            max_trials=config.trails,
            seed=42,
            directory=output_path,
            project_name=tuning_mode)
    # otherwise, we will be using the bayesian optimization tuner
    else:
        # instantiate the bayesian optimization tuner object
        print("[INFO] instantiating a bayesian optimization tuner object...")
        tuner = kt.BayesianOptimization(
            build_model,
            objective=config.objective,
            max_trials=config.trails,
            seed=42,
            directory=output_path,
            project_name=tuning_mode)

    hp_train_ds = train_ds

    # perform the hyperparameter search
    print("[INFO] performing hyperparameter search...in mode ", tuning_mode)
    # initialize an early stopping callback to prevent the model from
    # overfitting/spending too much time training with minimal gains
    es = EarlyStopping(
        monitor=config.es_monitor,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True)

    tuner.search(
        hp_train_ds,
        validation_data = val_ds,
        batch_size=config.BS,
        callbacks=[es],
        epochs=config.EPOCHS
    )
    # grab the best hyperparameters
    bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
    utils.print_bestparams(bestHP)
    return tuner

def loadmodel():
    model = None
    if os.path.exists(model_path):
        print("loaded saved best model")
        model = tf.keras.models.load_model(model_path)
    else:
        print("[INFO] training the best model...")
        if tuning_mode == "hyperband":
            # instantiate the hyperband tuner object
            print("[INFO] instantiating a hyperband tuner object...")
            tuner = kt.Hyperband(
                build_model,
                objective="sparse_categorical_accuracy",
                max_epochs=config.EPOCHS,
                factor=3,
                seed=42,
                directory=output_path,
                project_name=tuning_mode)
        # check if we will be using the random search tuner
        elif tuning_mode == "random":
            # instantiate the random search tuner object
            print("[INFO] instantiating a random search tuner object...")
            tuner = kt.RandomSearch(
                build_model,
                objective="sparse_categorical_accuracy",
                max_trials=10,
                seed=42,
                directory=output_path,
                project_name=tuning_mode)
        # otherwise, we will be using the bayesian optimization tuner
        else:
            # instantiate the bayesian optimization tuner object
            print("[INFO] instantiating a bayesian optimization tuner object...")
            tuner = kt.BayesianOptimization(
                build_model,
                objective="sparse_categorical_accuracy",
                max_trials=10,
                seed=42,
                directory=output_path,
                project_name=tuning_mode)

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        utils.print_bestparams(bestHP)

        model = tuner.hypermodel.build(bestHP)
    return model

def train(model):
    # Recreate the exact same model, including its weights and the optimizer

    # Setup TensorBoard, early stop, checkpoint callbacks.
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            model_checkpoint_dir, save_best_only=True),
        EarlyStopping(
            monitor="loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True)]

    H = model.fit(train_ds,validation_data =val_ds,
                  epochs=config.EPOCHS, callbacks=callbacks, verbose=1)
    model.save(model_path)


if mode == 'test':

    test_part = 2 if val_part == 1 else 1

    test_ds =None
    # taking less samples from dataset when no gpu is present
    gpu_available = tf.test.is_gpu_available()
    if not gpu_available or args['env'] == 'local':
        test_ds, file_count = get_ds_from_dataset([fold], part=test_part,take_count=localcount)
    else:
        test_ds, file_count = get_ds_from_dataset([fold],part=test_part)


    print("test_ds size=", tf.data.experimental.cardinality(test_ds))

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model(model_path)

    # Show the model architecture
    new_model.summary()

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = new_model.evaluate(test_ds)
    print("test loss, test acc:", results)

else:


    train_ds =None
    val_ds =None
    file_count =None
    #loading dataset
    if file_type == 'dataframe':

        folds = [i for i in range(1, 6)]
        folds.remove(fold)

        # taking less samples from dataset when no gpu is present
        gpu_available = tf.test.is_gpu_available()
        if not gpu_available or args['env'] == 'local':
            train_ds, file_count = get_ds_from_dataset(folds,take_count = localcount)
            val_ds, file_count = get_ds_from_dataset([fold],val_part,take_count= int(localcount / 2))
        else:
            train_ds, file_count = get_ds_from_dataset(folds)
            val_ds, file_count = get_ds_from_dataset([fold],val_part)
    else:
        train_ds, file_count = get_ds()

    for item in train_ds.take(1).as_numpy_iterator():
        print("printing first time train ds",item)

    for item in train_ds.take(1).as_numpy_iterator():
        print("printing second time train ds",item)

    for item in val_ds.take(1).as_numpy_iterator():
        print("printing first time val ds", item)

    for item in val_ds.take(1).as_numpy_iterator():
        print("printing second time val ds", item)


    print("val_ds size=", tf.data.experimental.cardinality(val_ds))
    print("train_ds size=", tf.data.experimental.cardinality(train_ds))


    #starting hyperparameter tuning if args['hpsearch'] is set to true
    bestHP = []
    #callling hp tuner and it will run if flag is set
    if mode=='tune':
        tuner = hp_tune()

    ##training model
    if mode=='train':
        model = loadmodel()
        train(model)
        # Recreate the exact same model, including its weights and the optimizer
        new_model = tf.keras.models.load_model(model_path)

        # Show the model architecture
        new_model.summary()

