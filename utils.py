import string
import numpy as np
alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)


def int2char(x):
    return alphabet_list[x]

def get_rand01():
    return np.random.choice([0,1], p=[0.25,0.75])


