from ineuron import config, Dataloader
import kerastuner as kt
from ineuron.model import build_model
import tensorflow as tf

def tune_hp(mode, root_dir, output_path,es,file_type,tuningcount,env):
    '''
    tunes model for various hyperparameters and returns tuner object
    :param mode: one of 'random', 'hyperband', 'bayesian_optimization'
    :param root_dir: data directory path
    :param output_path: path to save tuning logs
    :param es: early stopping callback function
    :return: tuner
    '''

    # check if we will be using the hyperband tuner
    if mode == "hyperband":
        # instantiate the hyperband tuner object
        print("[INFO] instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            build_model,
            objective="val_sparse_categorical_accuracy",
            max_epochs=config.EPOCHS,
            factor=3,
            seed=42,
            directory=output_path,
            project_name=mode)
    # check if we will be using the random search tuner
    elif mode == "random":
        # instantiate the random search tuner object
        print("[INFO] instantiating a random search tuner object...")
        tuner = kt.RandomSearch(
            build_model,
            objective="val_sparse_categorical_accuracy",
            max_trials=10,
            seed=42,
            directory=output_path,
            project_name=mode)
    # otherwise, we will be using the bayesian optimization tuner
    else:
        # instantiate the bayesian optimization tuner object
        print("[INFO] instantiating a bayesian optimization tuner object...")
        tuner = kt.BayesianOptimization(
            build_model,
            objective="val_sparse_categorical_accuracy",
            max_trials=10,
            seed=42,
            directory=output_path,
            project_name=mode)

    # if gpu available then use full dataset else take 1000 samples
    train_ds, val_ds, test_ds = Dataloader.get_ds(root_dir, 0.8, 0.1,file_type)
    # taking less samples from dataset when no gpu is present
    gpu_available = tf.test.is_gpu_available()
    if not gpu_available or env == 'local':
        trainds_count = tuningcount
        train_ds = train_ds.take(trainds_count)
        val_ds = val_ds.take(int(trainds_count / 10))

    #considering only 30% of train data
    count = tf.data.experimental.cardinality(train_ds).numpy()
    print('hyperparameter tuning train dataset size=',count)

    count_val = tf.data.experimental.cardinality(val_ds).numpy()
    print('hyperparameter tuning val dataset size=', count_val)

    # perform the hyperparameter search
    print("[INFO] performing hyperparameter search...in mode ",mode)
    tuner.search(
        train_ds,
        validation_data=val_ds,
        batch_size=config.BS,
        callbacks=[es],
        epochs=config.EPOCHS
    )
    return tuner