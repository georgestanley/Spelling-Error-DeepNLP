import matplotlib.pyplot as plt
import string
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus import stopwords
import numpy as np
import os, logging, datetime
import time
import torch
import argparse

alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
all_letters = string.ascii_letters + " .,;'"
np.random.seed(0)
torch.manual_seed(0)


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_in_log(log, save_step, scalar_dict=None, text_dict=None, image_dict=None, num_classes=1):
    if scalar_dict:
        [log.add_scalar(k, v, save_step) for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(k, v, save_step) for k, v in text_dict.items()]
    log.flush()


def int2char(x):
    """Given an integer between 0 and 25, returns the corresponding english alphabet of that position. Used in the
            insert_errors function"""

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
    return np.random.choice([1, 2, 3], p=[0.34, 0.33, 0.33])


def check_dir(path):
    '''Used to create directories to store training artifacts if  folder DNE.'''

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


def f1_score_manual(tn,fp,fn,tp):

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))
    return f1_score

