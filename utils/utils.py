import string
import numpy as np
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus import stopwords
import numpy as np
import os, logging, datetime

alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
all_letters = string.ascii_letters + " .,;'"


def int2char(x):
    return alphabet_list[x]

def get_rand01():
    return np.random.choice([0,1], p=[0.25,0.75])


def preapre_dataset():
    # worldlist10000 source: MIT

    #x = WordListCorpusReader('.', ['datasets/wordlist.10000.txt'])
    x = WordListCorpusReader('.', ['datasets/Oxford5000.txt'])
    print("Lenght =", len(x.words()))
    words = x.words()
    # print(words)
    stopwords_list = list(stopwords.words('english'))
    # print(stopwords_list)
    # blog_data.text = blog_data.text.apply(lambda t: ' '.join([words for words in t.split() if words not in stopwords]) )
    words = [word for word in words if word not in stopwords_list]*2
    # print(xx)

    # print(words)
    data_arr = np.array((words, np.ones(len(words),dtype=np.int))).T
    print(np.shape(data_arr))
    # print(data_arr)
    #x = np.repeat(data_arr, repeats=1)

    return data_arr


def insert_errors(data_arr):
    # generate random 0 or 1 of whether to generate errors in word input
    # if 1 , generate errors. assign higher prob for 1
    # Err type 1: replace one character by another
    # Err type 2: delete a character
    # Err type 3: Add an extra character
    x_temp = []
    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            x_temp.append(yy[0:rep_pos] + rep_char + yy[rep_pos + 1:])

    x_temp2=[]
    for x in data_arr[:,0]:
        x_temp2.append(x)

    '''
    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            # print("old word=",yy)
            # rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos] + yy[rep_pos + 1:])
            # print("new word=",yy)
            # x_temp.append(y)

    '''
    '''
    for x in data_arr[:, 0]:
        if get_rand01() == 1:
            yy = np.array2string(x).replace("'", "")
            #print("old word=", yy)
            rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos] + rep_char + yy[rep_pos:])
            # print("new word=",yy[0:rep_pos] + rep_char + yy[rep_pos:])
            # x_temp.append(y)


    '''

    x_temp_arr = np.array((x_temp, np.zeros(len(x_temp),dtype=np.int ))).T
    print("Shape of Error data", np.shape(x_temp_arr))
    data_arr = np.concatenate((data_arr, x_temp_arr))

    x_temp_arr2 = np.array((x_temp2, np.ones(len(x_temp2), dtype=np.int))).T
    data_arr = np.concatenate((data_arr, x_temp_arr2))


    print("Shape after adding new data", np.shape(data_arr))
    # print(data_arr)

    return data_arr

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_logger(logdir, name, evaluate=False):
    # Set logger for saving process experimental information
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logger.ts = ts
    if evaluate:
        file_path = os.path.join(logdir, "evaluate_{}.log".format(ts))
    else:
        file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    # strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr = logging.StreamHandler()
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    return logger