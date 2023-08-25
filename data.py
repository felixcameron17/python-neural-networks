import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(
        images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

# DISCLAIMER:
# This file was copied from:
# https://github.com/Bot-Academy/NeuralNetworkFromScratch/blob/master/data.py

# the 'data' folder, with the 'mnist.npz' file is also taken from the repository

# The whole project was also helped by watching this youtube video:
# https://www.youtube.com/watch?v=9RN2Wr8xvro

# I have no intention to use this code to profit or pass it off as my own.
# I am writing this code so that I can learn more about neural networks.
