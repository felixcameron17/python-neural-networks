from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
from del_guess import del_
import os
import config

epochs = config.epochs
imgs = config.imgs
learn_rate = config.learn_rate

# varible names conventions are as follows:
# 'w' means weight |'b' means bias | 'i' means input | 'h' means hidden | 'o' means output
# for example 'w_i_h' reads as: weight_input_hidden, or the weight value connecting the input to the hidden
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.05
nr_correct = 0
epoch_index = 0
img_index = 0
acc_str = ''

# an epoch is 1 iteration of the images being passed through the program
# the second nested loop is for each image
for epoch in range(epochs):
    epoch_index += 1
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)

        # forward propagation input --> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        # forward propagation hidden --> input
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # error calculation
        e = 1 / len(o) * np.sum((o - 1) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # back propagation output --> hidden
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # back propagation hidden --> input
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h = -learn_rate * delta_h

    # print accuracy for this epoch
    print(
        f"Epoch: {epoch_index}/{epochs} | Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}% ({nr_correct}/60000)")

    acc_str += f"Epoch: {epoch_index}/{epochs} | Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}% ({nr_correct}/60000)\n"
    nr_correct = 0

del_()

for i in range(imgs):
    # show results
    index = randint(0, 59999)
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap='Greys')

    img.shape += (1,)

    # forward propagation input --> hidden
    h_pre = b_i_h + w_i_h @ img
    h = 1 / (1 + np.exp(-h_pre))

    # forward propagation hidden --> input
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'guesses/{o.argmax()}/{index}.png')
    os.system('cls')
    print(f'{acc_str}{i+1}/{imgs} images generated')


# https://github.com/felixcameron17
