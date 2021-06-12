#soruce: https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

# open text file and read in data as `text`

def prepare_data():
    with open('datasets/news.en.heldout-00000-of-00050', 'r',encoding='utf-8') as f:
        text = f.read()
    #print(text[:300])

    text = "sstanley is my name. i come from the world. i am not an alien."

    chars = tuple(set(text))
    #print(len(chars))
    int2char = dict(enumerate(chars))
    #print(int2char)
    char2int = {ch: ii for ii, ch in int2char.items()}
    #print(char2int)

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])
    print(encoded)


    decoded = np.array([int2char[ii] for ii in encoded])
    print(decoded)

    return  encoded




def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # print("1.one_hot shape=", one_hot.shape)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

encoded_arr = prepare_data()
#test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(np.array([encoded_arr[:8]]), 30)
one_hot = np.squeeze(one_hot)
print("one hot vec shape=", one_hot.shape)
print(one_hot)

# task preapre dataset of words and labels
# shape [word, label]




