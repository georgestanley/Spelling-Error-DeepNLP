# Tasks
# 1. Prepare a dataset of [words,label]
# 2. Insert errors into the dataset
# 3. Split as Train and Valid
# 4. Prepare an MLP to classify it as Correct OR Wrong spelling
# Extra:
# 5. Convert MLP to CNN
# 6. Experiment with word corpus
# 7. Implement it as docker.

# source : https://www.geeksforgeeks.org/nlp-wordlist-corpus/

from nltk.corpus.reader import WordListCorpusReader
import numpy as np
from nltk.corpus import stopwords
from utils import *


def preapre_dataset():
    # worldlist10000 source: MIT

    x = WordListCorpusReader('.', ['datasets/wordlist.10000.txt'])
    print("Lenght =", len(x.words()))
    words = x.words()
    # print(words)
    stopwords_list = list(stopwords.words('english'))
    # print(stopwords_list)
    # blog_data.text = blog_data.text.apply(lambda t: ' '.join([words for words in t.split() if words not in stopwords]) )
    words = [word for word in words if word not in stopwords_list]
    # print(xx)

    # print(words)
    data_arr = np.array((words, np.ones(len(words)))).T
    print(np.shape(data_arr))
    # print(data_arr)

    return data_arr


def insert_errors(data_arr):
    # generate random 0 or 1 of whether to generate errors in word input
    # if 1 , generate errors. assign higher prob for 1
    # Err type 1: replace one character by another
    # Err type 2: delete a character
    # Err type 3 : Add an extra character
    x_temp = []
    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            # print("old word=",yy)
            rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos] + rep_char + yy[rep_pos + 1:])
            # print("new word=",yy)
            # x_temp.append(y)

    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            # print("old word=",yy)
            #rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos]+ yy[rep_pos + 1:])
            # print("new word=",yy)
            # x_temp.append(y)

    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            print("old word=",yy)
            rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos] + rep_char + yy[rep_pos:])
            #print("new word=",yy[0:rep_pos] + rep_char + yy[rep_pos:])
            # x_temp.append(y)

    x_temp_arr = np.array((x_temp, np.zeros(len(x_temp)))).T
    print("Shape of Error data", np.shape(x_temp_arr))
    data_arr = np.concatenate((data_arr, x_temp_arr))
    print("Shape after adding new data", np.shape(data_arr))
    # print(data_arr)

    return data_arr



data_arr = preapre_dataset()
data_arr = insert_errors(data_arr)

#Create an MLP networl
#Create a CNN network
#compare performance of both

