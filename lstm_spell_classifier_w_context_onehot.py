import collections

import string, argparse, json, os, re
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from Model import LSTMModel
import sys
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.utils import get_rand01, check_dir, int2char, get_logger, plot_graphs, save_in_log, get_rand123
# import wandb
from sklearn.metrics import f1_score
from datetime import datetime

import time

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_iters = 100000
print_every = 1000
plot_every = 1000
batchsize = 100
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_# "
alph_len = len(alph)

exp_id = datetime.now().strftime('%Y%m%d%H%M%S')


# maxlen : dev10 : 66
# maxlen : development_documents = 174


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help="folder containing the data")
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--input_file', type=str, default='dev_10.jsonl')
    parser.add_argument('--val_file', type=str , default='dev_10.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--optim', type=str, default="Adam", help="optimizer to use")
    parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden layer Dim')
    parser.add_argument('--hidden_layers', type=int, default=2, help='the number of hidden LSTM layers')
    parser.add_argument('--maxlen', type=int, default=60, help='the max length of words in a single seq')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()
    hparam_keys = ["lr", "bs", "optim", "hidden_dim", "hidden_layers"]  # changed from loss to size
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'lstm_context_onehot', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "{}_models".format(exp_id)))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def get_wikipedia_text(file_name):
    '''
    Returns a pandas Dataframe containing the extracted texts.
    :param file_name:
    :return:
    '''
    data = []
    with open(file_name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            data.append(json.loads(line)['text'].lower())
        data = np.array(data)
    return data


# @timeit
def remove_punctuation(texts):
    '''

    :param text: String
    :return: ans: String
    '''
    ans = ""
    stripPunct = str.maketrans('', '', string.punctuation)
    new = np.array([i.translate(stripPunct) for i in texts])
    return new


# @timeit
def cleanup_data(data):
    """
    :param: data :Pandas dataframe [1 column]
    :returns data : Pandas Dataframe [1 column]
    """
    # data['text'] = data['text'].apply(lambda x: remove_punctuation(x))
    f = lambda x: remove_punctuation(x)
    data = f(data)
    return data


def generate_N_grams(data, ngram=5):
    """
    Takes and input a Dataframe of texts.Breaks it into list of 5-grams inside a Dataframe
    # label meanings:
    # 0: no error in middle word
    # 1: With error in middle word

    """

    new_dataset = []
    lens = []
    for n, text in tqdm(enumerate(data)):
        # TODO https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python

        r = r'\S*\d+\S*'  # Remove alpha-num words ; https://stackoverflow.com/a/65105960/5959601
        text = re.sub(r, '', text)
        text = text.split()
        text[:] = [tup for tup in text if tup.isalpha()]
        text[:] = [tup for tup in text if tup.isascii()]

        for i in range(0, len(text) - ngram + 1):
            x = []
            for j in range(5):
                x.append(text[i + j])
            # new_dataset.append([x])
            chars = (' '.join(x))
            lens.append(len(chars))
            new_dataset.append(chars)

    new_dataset = np.array(new_dataset)
    labels = np.zeros(len(new_dataset))
    # new_dataset : list(dataset_len) ;e.g.  'big brother nineteen eightyfour big'

    return new_dataset, labels  # new_dataset:


def convert_to_pytorch_dataset(train_data, val_data):
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=1, pin_memory=True
                                  )

    val_dataset = MyDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True,
                                # collate_fn=collate_fn
                                )

    return train_dataloader, val_dataloader


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.words = data[0]
        self.labels = data[1]

    def __getitem__(self, i):
        # x = self.words
        word = self.words[i]
        label = int(self.labels[i])
        return (word, label)

    def __len__(self):
        return len(self.labels)


def initialize_model():
    input_dim = len(alph)
    hidden_dim = args.hidden_dim  # TODO : Iterate over different hidden dim sizes
    layer_dim = args.hidden_layers
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, criterion, optimizer


def insert_errors(data):  #
    '''

    :param data: ndarray (batch_size,2)
    :return: data : ndarray ( ?? ,2)
    '''

    # print('data shape before ', np.shape(data))
    temp = []
    for i, x in enumerate(data):
        switch_val = get_rand123()
        if switch_val == 1:
            if get_rand01() == 1:
                # Type 1: Replace a character
                x = x.replace("'", "")
                x = x.split()
                yy = x[2]
                # yy = np.array2string(x).replace("'", "")
                rep_char = int2char(np.random.randint(0, 26))
                rep_pos = np.random.randint(low=0, high=len(yy))
                false_word = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
                x[2] = false_word
                temp.append(' '.join(x))  # [i][2] = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
        elif switch_val == 2:
            if get_rand01() == 1 and len(x) > 1:
                # Type 2: delete a character
                x = x.replace("'", "").split()
                yy = x[2]
                rep_pos = np.random.randint(low=0, high=len(yy))
                false_word = yy[0:rep_pos] + yy[rep_pos + 1:]
                x[2] = false_word
                temp.append(' '.join(x))
        elif switch_val == 3:
            pass
    label_true = [0] * len(data)
    label_false = [1] * len(temp)
    labels = label_true + label_false
    data = data + temp
    return data, labels


def one_hot_encode_data(new_dataset, labels):
    maxlen = args.maxlen
    new_dataset, labels = insert_errors(new_dataset)
    arr_len = []
    for n in range(len(new_dataset)):
        input_seq = ''
        new_dataset[n] = new_dataset[n][:maxlen]
        arr_len.append(len(new_dataset[n])-1)
        new_dataset[n] = new_dataset[n].ljust(maxlen, '*')
        # for i in range(len(new_dataset)):
        xx = [alph.index(character) for character in new_dataset[n]]
        new_dataset[n] = xx  # new_dataset : list(dataset_len) ; e.g. [[27, 34, 32, 76, 21, ... ], [28,25,25,..]]

    new_dataset = np.array(new_dataset)
    new_dataset = torch.from_numpy(new_dataset)
    new_dataset = torch.nn.functional.one_hot(new_dataset.to(torch.int64), num_classes=77)
    labels = torch.from_numpy(np.array(labels))

    #shuffle the batch
    r = torch.randperm(new_dataset.size()[0])
    new_dataset = new_dataset[r]
    labels = labels[r]

    return new_dataset, labels, arr_len


def train_model(train_loader, model, criterion, optim, writer, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    correct = 0
    # model.train()

    for i, data in enumerate(tqdm(train_loader)):
        X_vec, Y_vec, sentence_length = one_hot_encode_data(new_dataset=data[0], labels=data[1])
        X_vec = X_vec.type(torch.FloatTensor).to(device)
        Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)
        optim.zero_grad()
        outputs = model(X_vec,sentence_length)  # (n_words, 2)#
        loss = criterion(outputs, Y_vec)
        ssg, predicted = torch.max(outputs.data, 1)
        correct += (predicted == Y_vec).sum()
        # c = collections.Counter(predicted.cpu().detach().numpy())
        # print("Input Distribution",c)
        loss.backward()
        optim.step()
        total += Y_vec.size(0)

        batch_size = Y_vec.size(0)
        total_loss += loss.item()

        # mean_val_loss = total_loss / total
    alpha = (len(train_loader.dataset)) / batch_size
    mean_train_loss = total_loss / alpha
    mean_train_accuracy = 100 * correct / total
    scalar_dict = {'Loss/train': mean_train_loss, 'Accuracy/train': mean_train_accuracy}
    print(f"mean_train_loss:{mean_train_loss} mean_train_acc;{mean_train_accuracy}")
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_train_loss, mean_train_accuracy


def val_model(val_loader, model, criterion, logger, writer, epoch):
    # TODO: Improve this validation section
    correct = 0
    total = 0
    f1 = 0

    total_loss = 0
    total_accuracy = 0
    total = 0
    # model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            #data = next(iter(val_loader))
            X_vec, Y_vec , sentence_length = one_hot_encode_data(new_dataset=list(data[0]), labels=data[1])
            X_vec = X_vec.type(torch.FloatTensor).to(device)
            Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)

            outputs = model(X_vec, sentence_length)  # (n_words, 2)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            total += Y_vec.size(0)
            loss = criterion(outputs, Y_vec)
            correct += (predicted == Y_vec).sum()

            f1 = f1_score(predicted.cpu(), Y_vec.cpu())
            c = collections.Counter(predicted.cpu().detach().numpy())
            print(c)
            batch_size = Y_vec.size(0)
            total_loss += loss.item()

            # temp_to_print = np.column_stack((X_token, Y_vec.cpu(), predicted.cpu()))
            # to_print = np.row_stack((to_print, temp_to_print))

        # to_print = pd.DataFrame(to_print)
        # to_print.to_csv('data2.csv')
        # mean_val_loss = total_loss / total
        alpha = (len(val_loader.dataset)) / batch_size
        # alpha = 1000 / batch_size
        mean_val_loss = total_loss / alpha
        mean_val_accuracy = 100 * correct / total
        scalar_dict = {'Loss/val': mean_val_loss, 'Accuracy/val': mean_val_accuracy}
        print(f"mean_val_loss:{mean_val_loss} mean_val_acc:{mean_val_accuracy} , f1_score={f1},total_correct={correct},"
          f"total_samples={total}")
    #save_in_log(writer, epoch, scalar_dict=scalar_dict)
    # accuracy = 100 * correct / total
    # print(f" Word = {X_token[600]} Prediction= {predicted[600]} loss = {loss.item()} accuracy= {accuracy} f1_Score={f1}")
    return mean_val_loss, mean_val_accuracy.cpu(), f1


def main(args):
    writer = SummaryWriter()
    logger = get_logger(args.output_folder, args.exp_name)
    train_data = get_wikipedia_text(os.path.join(args.data_folder, args.input_file))
    train_data = cleanup_data(train_data)
    train_data = generate_N_grams(train_data)
    val_data = get_wikipedia_text(os.path.join(args.data_folder, args.val_file))
    val_data = cleanup_data(val_data)
    val_data = generate_N_grams(val_data)
    # data = one_hot_encode_data(new_dataset = data[0], labels = data[1])
    # data = convert_to_numpy(data)
    # dataz = np.load('data\\5_gram_dataset.npz')
    # dataz = np.load(os.path.join(args.data_folder, args.input_file))
    # data = (dataz['arr_0'], dataz['arr_1'])
    train_loader, val_loader = convert_to_pytorch_dataset(train_data,val_data)
    model, criterion, optim = initialize_model()

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])

    print("Dataset size: {} samples".format(len(train_loader.dataset)))  # TODO
    logger.info(expdata)
    logger.info('train_data {}'.format(train_loader.dataset.__len__()))  # TODO
    logger.info('val_data {}'.format(val_loader.dataset.__len__()))  # TODO

    n_epoch = args.epochs

    # test_dataloader(train_loader)
    train_losses, val_losses, val_accuracies, val_f1s = [0.0], [0.0], [0.0], [0.0]
    for epoch in range(n_epoch):

        train_loss, train_acc = train_model(train_loader, model, criterion, optim, writer, epoch)
        val_loss, val_acc, val_f1 = val_model(val_loader, model, criterion, logger, writer, epoch)

        logger.info(f'Epoch{epoch}')
        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))
        logger.info('Validation F1: {}'.format(val_f1))

        if val_f1 > max(val_f1s) or val_acc > max(val_accuracies):
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best_{}.pth".format(epoch)))
            logger.info('Model Saved')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

    plot_graphs(n_epoch, args.model_folder, logger, train_losses, val_losses, val_accuracies, val_f1s)

    return


if __name__ == "__main__":
    start = datetime.now()
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device}")
    print("LSTM Spelling Classifier with context -- One-Hot")
    print(vars(args))
    print()
    main(args)
    print(datetime.now() - start)
