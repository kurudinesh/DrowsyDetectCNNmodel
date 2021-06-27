import os
# define the path to our output directory
modelname = "DD2CNN_model.h5"

# initialize the input shape and number of classes
INPUT_SHAPE = (468, 3)
NUM_CLASSES = 3
# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS = 5
BS = 32
EARLY_STOPPING_PATIENCE = 3