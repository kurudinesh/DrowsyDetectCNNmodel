# set the matplotlib backend so figures can be saved in the background
import matplotlib
import numpy as np
matplotlib.use("Agg")
# import the necessary package
import matplotlib.pyplot as plt


def save_plot(H, path):
    '''
    plots train and val loss
    :param H: history object returned after fitting model
    :param path: path for saving plot as image
    '''
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["sparse_categorical_accuracy"], label="train_acc")
    plt.plot(H.history["val_sparse_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(path)

def print_bestparams(bestHP):
    '''
    prints best values sampled from hyper parameter tuning
    :param bestHP:
    :return:
    '''
    print("[INFO] optimal number of filters in conv_1 layer: {}".format(
        bestHP.get("filter_1")))
    print("[INFO] optimal size of kernel in conv_1 layer: {}".format(
        bestHP.get("kernel_1")))
    print("[INFO] optimal alpha in leaky_relu_1 layer: {}".format(
        bestHP.get("leaky_alpha_1")))
    # print("[INFO] optimal pool size of maxpool_1 layer: {}".format(
    #     bestHP.get("pool_1")))
    print("[INFO] optimal dropout rate of dropout_1 layer: {}".format(
        bestHP.get("dropout_1")))

    print("[INFO] optimal number of filters in conv_x layer: {}".format(
        bestHP.get("filter_x")))
    print("[INFO] optimal size of kernel in conv_x layer: {}".format(
        bestHP.get("kernel_x")))
    print("[INFO] optimal alpha in leaky_relu_x layer: {}".format(
        bestHP.get("leaky_alpha_x")))
    # print("[INFO] optimal pool size of maxpool_x layer: {}".format(
    #     bestHP.get("pool_x")))
    print("[INFO] optimal dropout rate of dropout_x layer: {}".format(
        bestHP.get("dropout_x")))

    print("[INFO] optimal learning rate: {:.4f}".format(
        bestHP.get("learning_rate")))

def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
