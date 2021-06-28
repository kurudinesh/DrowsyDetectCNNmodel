# import the necessary packages
from ineuron import config
from ineuron import utils
from ineuron import Dataloader
from ineuron.hp_tuner import tune_hp
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import tensorflow as tf
import os
import pickle as pk

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from ineuron.Dataloader import print_sample

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
ap.add_argument("-l", "--localcount", required=False, type=int,
                default=1000,
                help="in local mode the length of train_ds =batchsize*localcount,\n"
                        +" if count is -1 then whole data will be loaded, Default 1000",
                )
ap.add_argument("-tc", "--tuningcount", required=False, type=int,
                default=1000,
                help="in local mode the length of val_ds =batchsize*tuningcount,\n"
                        +" if count is -1 then whole data will be loaded,  Default 1000",
                )
ap.add_argument("-f", "--format", required=False, type=str,
                choices=["csv", "npy"],
                default='csv',
                help="file format landmarks of video frames are saved in data directory, DEFAULT 'csv'"
                )
ap.add_argument("-s", "--hpsearch", required=False, type=bool,
                choices=[True, False],
                default=False,
                help="enter true for hyperparameter tuning, default False"
                )

args = vars(ap.parse_args())

#setting output_base_folder_path
output_path = args['output']
tuning_mode = args["tuner"]
localcount = args['localcount']
tuningcount = args['tuningcount']
file_type = args['format']
env = args['env']

#setting path for saving plot of training loss, accuracy
plot_path = os.path.join(output_path,tuning_mode+'_plot.png')


#setting logs, checkpoints, data drirectories
tensorboard_log_dir = os.path.join(output_path, 'logs')
model_checkpoint_dir = os.path.join(output_path,"checkpoints")
model_path = os.path.join(output_path,config.modelname)

#generating dataset from landmark .npy files of video frames
root_dir = args['data']

# initialize an early stopping callback to prevent the model from
# overfitting/spending too much time training with minimal gains
es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

#tuning the model with given hyper parameters\
if args['hpsearch']:
    tuner = tune_hp(tuning_mode,root_dir,output_path,es,file_type,tuningcount,env)
    # grab the best hyperparameters
    bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
    utils.print_bestparams(bestHP)


#if gpu available then use full dataset else take 1000 samples
train_ds, val_ds, test_ds = Dataloader.get_ds(root_dir,0.8,0.1,file_type)


#taking less samples from dataset when no gpu is present
gpu_available = tf.test.is_gpu_available()
if not gpu_available or args['env']=='local':
    trainds_count = localcount
    train_ds = train_ds.take(trainds_count)
    val_ds = val_ds.take(int(trainds_count / 10))
    test_ds = test_ds.take(int(trainds_count / 10))

print('best model training dataset sample counts')
print_sample(None,train_ds,val_ds,test_ds,1)

# build the best model and train it

# Recreate the exact same model, including its weights and the optimizer
model = None
if os.path.exists(model_path):
    print("loaded saved best model")
    model = tf.keras.models.load_model(model_path)
else:
    print("[INFO] training the best model...")
    model = tuner.hypermodel.build(bestHP)

# Setup TensorBoard, early stop, checkpoint callbacks.
callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            model_checkpoint_dir, save_best_only=True),
        es]


H = model.fit(train_ds,
	validation_data=test_ds, batch_size=config.BS,
	epochs=config.EPOCHS, callbacks=callbacks, verbose=1)


model.save(model_path)

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model(model_path)

# Show the model architecture
new_model.summary()

# evaluate the network
print("[INFO] evaluating network...")
# evaluate model
_, accuracy = new_model.evaluate(test_ds, batch_size=config.BS, verbose=0)

# summarize scores
utils.summarize_results(accuracy)

#plot training history
utils.save_plot(H, plot_path)

