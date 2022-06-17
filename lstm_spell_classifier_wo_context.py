import pandas as pd
import string, argparse, json, os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from Model import MLPNetwork, RNN, LSTMModel
import sys, random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.utils import get_rand01, check_dir, int2char, get_logger, get_rand123, accuracy, save_in_log
# import wandb
from sklearn.metrics import f1_score
from datetime import datetime

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_iters = 100000
print_every = 1000
plot_every = 1000
batchsize = 100
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
exp_id = datetime.now().strftime('%Y%m%d%H%M%S')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--optim', type=str, default="Adam", help="optimizer to use")
    parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden layer Dim')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()
    hparam_keys = ["lr", "bs", "optim"]  # changed from loss to size
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'lstm_noncontext', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "{}_models".format(exp_id)))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def insert_errors(data):  #
    '''

    :param data: ndarray (batch_size,2)
    :return: data : ndarray ( ?? ,2)
    '''
    # print('data shape before ',np.shape(data))
    temp = []
    for x in data[:, 0]:
        switch_val = get_rand123()
        if switch_val == 1:
            if get_rand01() == 1:
                # Type 1: Replace a character
                yy = np.array2string(x).replace("'", "")
                rep_char = int2char(np.random.randint(0, 26))
                rep_pos = np.random.randint(low=0, high=len(yy))
                temp.append(yy[0:rep_pos] + rep_char + yy[rep_pos + 1:])
        elif switch_val == 2:
            if get_rand01() == 1 and len(x) > 1:
                # Type 2: delete a character
                yy = np.array2string(x).replace("'", "")
                rep_pos = np.random.randint(low=0, high=len(yy))
                temp.append(yy[0:rep_pos] + yy[rep_pos + 1:])
        elif switch_val == 3:
            # type 3
            pass

    x2 = np.ones((len(temp)))
    x = np.column_stack((temp, x2))
    data = np.concatenate((data, x))
    # print('data shape after ',np.shape(data))
    return data


def binarize(token, label, alph):
    bin_beg = [0] * len(alph)
    bin_middle = [0] * len(alph)
    bin_end = [0] * len(alph)

    bin_beg[alph.index(token[0])] += 1
    bin_end[alph.index(token[-1])] += 1

    for i in range(1, len(token) - 1):
        bin_middle[alph.index(token[i])] += 1

    bin_all = bin_beg + bin_middle + bin_end
    return torch.tensor(bin_all), torch.tensor(int(float(label))), token


def vectorize_data(data_arr,with_error):
    # https://arxiv.org/pdf/1608.02214.pdf
    '''

    :param data_arr:list(2). [tuple(200),tuple(200)]
    :return:
    '''
    data_arr = np.column_stack((data_arr[0], data_arr[1]))  # ndarray (batch_size,2)
    if with_error:
        data_arr = insert_errors(data_arr)
    # X_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, len(alph) * 3))
    X_vec = torch.zeros((len(data_arr), len(alph) * 3))
    Y_vec = torch.zeros((len(data_arr), 1))
    X_token = []
    # TODO:
    # make X_token as np.array so that it can contain strings
    # shuflle it same way as for tensor ... X_token = X_token[r]

    for m, mini_batch_tokens in enumerate(zip(*[iter(data_arr)])):
        X_token_m = []
        x_mini_batch = torch.zeros((1, len(alph) * 3))  # (1,228)
        y_mini_batch = torch.zeros((1, 1))  # (1,1)

        for j, token in enumerate(mini_batch_tokens):
            x, y, z = binarize(token[0], token[1], alph)
            x_mini_batch, y_mini_batch, x_token = x, y, z
            '''
            if jumble_type == 'NO':
                x_mini_batch[j], x_token = binarize.noise_char(token, noise_type, alph)
            else:
                x_mini_batch[j], x_token = binarize.jumble_char(token, jumble_type, alph)
            '''
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)

        sys.stdout.flush()

    # print(X_vec.shape,Y_vec.shape)

    r = torch.randperm(X_vec.size()[0])
    X_vec = X_vec[r]
    Y_vec = Y_vec[r]

    X_token = np.asarray(X_token)
    X_token = X_token[r]

    return X_vec, Y_vec, X_token


def get_wikipedia_words(file_name):
    with open(file_name) as f:
        data = f.read()
        words = json.loads(data)
    return words


def convert_to_numpy(words):
    '''
    takes in a dictionary of words and converts it into and ndarray. Also outputs another ndarray of Zeroes for labels.
    :param words: dict
    :return: tuple(x1,x2); x1 ->ndarray , x2 -> ndarray
    '''
    non_ascii_keys = []
    for x in words.keys():
        if x.isascii() != True:
            non_ascii_keys.append(x)
    for x in non_ascii_keys:
        del words[x]

    x1 = np.array(list(words.keys()))
    x2 = np.zeros(x1.size)
    x = np.column_stack((x1, x2))
    return (x1, x2)


def convert_to_numpy_valdata(words):
    '''
    takes in a dictionary of words and converts it into and ndarray. Also outputs another ndarray of Zeroes for labels.
    :param words: dict
    :return: tuple(x1,x2); x1 ->ndarray , x2 -> ndarray
    '''
    non_ascii_keys = []
    for x in words.keys():
        if x.isascii() != True:
            non_ascii_keys.append(x)
    for x in non_ascii_keys:
        del words[x]

    x1 = np.array(list(words.keys()))
    # x2 = np.zeros(x1.size)
    x2 = np.array(list(words.values()))
    x = np.column_stack((x1, x2))
    return (x1, x2)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __getitem__(self, i):
        word = self.words[i]
        label = int(self.labels[i])
        return (word, label)

    def __len__(self):
        return len(self.labels)


def convert_to_pytorch_dataset(train_data, val_data):
    '''

    :param data: tuple (2)
    words : ndarray (9990,)
    labels: ndarray (9990,)
    :return:
    '''
    words = train_data[0]
    labels = train_data[1]
    val_words, val_labels = val_data[0], val_data[1]

    my_dataset = MyDataset(words, labels)
    my_dataloader = DataLoader(my_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = MyDataset(val_words, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    return my_dataloader, val_dataloader


def test_dataloader(my_dataloader):
    for i, (word, label) in enumerate(my_dataloader):
        print(i, word, label)
        return


def initialize_model(n_hidden_layers):
    input_dim = 228
    hidden_dim = args.hidden_dim  # TODO : Iterate over different hidden dim sizes
    layer_dim = n_hidden_layers
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()

    return model, criterion, optimizer


def train_model(train_loader, model, criterion, optim, epoch):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        X_vec, Y_vec, X_token = vectorize_data(data, with_error=True)  # xx shape:
        X_vec = torch.unsqueeze(X_vec, 1).requires_grad_().to(device)  # (n_words,228) --> (n_words , 1, 228)
        Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)
        optim.zero_grad()
        outputs = model(X_vec)  # (n_words, 2)#
        loss = criterion(outputs, Y_vec)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        batch_size = Y_vec.size(0)

    alpha = (len(train_loader.dataset))/batch_size
    loss /= alpha

    return loss


def val_model(val_loader, model, criterion, logger,epoch,writer ):
    correct = 0
    f1 = 0
    total_loss = 0
    total = 0
    to_print=np.empty((1,3))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, X_token = vectorize_data(data, with_error=False)  # xx shape:
            X_vec = torch.unsqueeze(X_vec, 1).requires_grad_().to(device)  # (n_words,228) --> (n_words , 1, 228)
            Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)

            outputs = model(X_vec)  # (n_words, 2)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            total += Y_vec.size(0)
            loss = criterion(outputs, Y_vec)
            correct += (predicted == Y_vec).sum()

            f1 = f1_score(predicted.cpu(), Y_vec.cpu())
            batch_size = Y_vec.size(0)
            total_loss += loss.item()
            temp_to_print = np.column_stack((X_token,Y_vec.cpu(),predicted.cpu()))
            to_print = np.row_stack((to_print,temp_to_print))

    to_print = pd.DataFrame(to_print)
    to_print.to_csv('data2.csv')
    mean_val_loss = total_loss / total
    mean_val_accuracy = 100 * correct / total
    scalar_dict = {'Loss/val': mean_val_loss, 'Accuracy/val': mean_val_accuracy}
    print(f"mean_val_loss:{mean_val_loss} mean_val_acc:{mean_val_accuracy} , f1_score={f1},total_correct={correct},"
          f"total_samples={total}")
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy, f1


def main(args):
    writer = SummaryWriter()
    args.hidden_dim = 256
    logger = get_logger(args.output_folder, args.exp_name)
    logger.info("Error 1 and 2")

    train_data = get_wikipedia_words(os.path.join(args.data_folder, "top_30000_words_over_200000.json"))
    train_data = convert_to_numpy(train_data)

    val_data = get_wikipedia_words(os.path.join(args.data_folder, "val_set_with_error.json"))
    val_data = convert_to_numpy_valdata(val_data)

    train_loader, val_loader = convert_to_pytorch_dataset(train_data, val_data)
    model, criterion, optim = initialize_model(n_hidden_layers=1)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])

    print("Dataset size: {} samples".format(len(train_loader.dataset)))  # TODO
    logger.info(expdata)
    logger.info('train_data {}'.format(train_loader.dataset.__len__()))  # TODO
    logger.info('val_data {}'.format(val_loader.dataset.__len__()))  # TODO

    n_epoch = 30

    train_losses, val_losses, val_accuracies, val_f1s = [0.0], [0.0], [0.0], [0.0]
    for epoch in range(n_epoch):
        train_loss = train_model(train_loader, model, criterion, optim, epoch)
        val_loss, val_acc, val_f1 = val_model(val_loader, model, criterion, logger,epoch, writer)

        logger.info(f'Epoch{epoch}')
        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        if val_f1 > max(val_f1s) or val_acc > max(val_accuracies):
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best_{}.pth".format(epoch)))
            logger.info('Model Saved')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

    plt.plot(np.arange(n_epoch + 1), train_losses)
    plt.title('Train Loss')
    plt.savefig(fname=os.path.join(args.model_folder, "plot_train_loss.png"))
    plt.show()

    plt.plot(np.arange(n_epoch + 1), val_losses)
    plt.title('Val Loss')
    plt.savefig(fname=os.path.join(args.model_folder, "plot_val_loss.png"))
    plt.show()

    plt.plot(np.arange(n_epoch + 1), val_accuracies)
    plt.title('Val Acc')
    plt.savefig(fname=os.path.join(args.model_folder, "plot_val_acc.png"))
    plt.show()

    plt.plot(np.arange(n_epoch + 1), val_f1s)
    plt.title('Val F1')
    plt.savefig(fname=os.path.join(args.model_folder, "plot_val_f1.png"))
    plt.show()

    return


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("LSTM Spelling Classifier")
    print(vars(args))
    print()
    main(args)
