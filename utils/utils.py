import matplotlib.pyplot as plt
import string
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus import stopwords
import numpy as np
import os, logging, datetime
import time
import torch

alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
all_letters = string.ascii_letters + " .,;'"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_in_log(log, save_step, scalar_dict=None, text_dict=None, image_dict=None, num_classes=1):
    if scalar_dict:
        [log.add_scalar(k, v, save_step) for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(k, v, save_step) for k, v in text_dict.items()]
    log.flush()


def int2char(x):
    return alphabet_list[x]


def get_rand01():
    '''
    Random choice of whether to generate error or not.
    0 -> Do not generate error
    1 -> Generate error word
    '''
    return np.random.choice([0, 1], p=[0.10, 0.90])


def get_rand123():
    '''
    Random choice of Error type to apply on selected string
    '''
    return np.random.choice([1, 2, 3], p=[0.50, 0.50, 0])


def preapre_dataset():
    # worldlist10000 source: MIT

    # x = WordListCorpusReader('.', ['datasets/wordlist.10000.txt'])
    x = WordListCorpusReader('.', ['datasets/Oxford5000.txt'])
    print("Lenght =", len(x.words()))
    words = x.words()
    # print(words)
    stopwords_list = list(stopwords.words('english'))
    # print(stopwords_list)
    # blog_data.text = blog_data.text.apply(lambda t: ' '.join([words for words in t.split() if words not in stopwords]) )
    words = [word for word in words if word not in stopwords_list] * 2
    # print(xx)

    # print(words)
    data_arr = np.array((words, np.ones(len(words), dtype=np.int))).T
    print(np.shape(data_arr))
    # print(data_arr)
    # x = np.repeat(data_arr, repeats=1)

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

    x_temp2 = []
    for x in data_arr[:, 0]:
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

    x_temp_arr = np.array((x_temp, np.zeros(len(x_temp), dtype=np.int))).T
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


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def plot_graphs(n_epoch, model_folder, logger, train_losses, val_losses, val_accuracies, val_f1s):
    # create plot
    plt.ion()
    train_losses.pop(0)
    plt.plot(np.arange(n_epoch), train_losses)
    logger.info('train loss: {}'.format(train_losses))
    plt.title('Train Loss')
    plt.ioff()
    plt.savefig(fname=os.path.join(model_folder, "plot_train_loss.png"))

    # plt.show()

    plt.ion()
    val_losses.pop(0)
    plt.plot(np.arange(n_epoch), val_losses)
    logger.info('val loss: {}'.format(val_losses))
    plt.title('Val Loss')
    plt.ioff()
    plt.savefig(fname=os.path.join(model_folder, "plot_val_loss.png"))
    # plt.show()

    plt.ion()
    val_accuracies.pop(0)
    plt.plot(np.arange(n_epoch), val_accuracies)
    logger.info('val acc: {}'.format(val_accuracies))
    plt.title('Val Acc')
    plt.ioff()
    plt.savefig(fname=os.path.join(model_folder, "plot_val_acc.png"))
    # plt.show()

    plt.ion()
    val_f1s.pop(0)
    plt.plot(np.arange(n_epoch), val_f1s)
    logger.info('val f1s: {}'.format(val_f1s))
    plt.title('Val F1')
    plt.ioff()
    plt.savefig(fname=os.path.join(model_folder, "plot_val_f1.png"))
