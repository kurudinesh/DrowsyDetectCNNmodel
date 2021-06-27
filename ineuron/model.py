# import the necessary packages
from . import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.metrics import SparseCategoricalAccuracy

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