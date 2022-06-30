import collections

import pandas as pd
import string, argparse, json, os, re
import numpy as np
import torch
from torch import nn
from Model import LSTMModel
import sys
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.utils import get_rand01, check_dir, int2char, get_logger, plot_graphs, save_in_log, get_rand123
from sklearn.metrics import f1_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_iters = 100000
print_every = 1000
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
alph_len = len(alph)

exp_id = datetime.now().strftime('%Y%m%d%H%M%S')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='data', help="folder containing the data")
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--input_file', type=str, default='dev_10.jsonl')
    parser.add_argument('--val_file', type=str, default='dev_10.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=1000, help='batch_size')
    parser.add_argument('--optim', type=str, default="Adam", help="optimizer to use")
    parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden layer Dim')
    parser.add_argument('--hidden_layers', type=int, default=2, help='the number of hidden LSTM layers')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()
    hparam_keys = ["lr", "bs", "optim", "hidden_dim", "hidden_layers"]  # changed from loss to size
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'lstm_context', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "{}_models".format(exp_id)))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def get_wikipedia_text(file_name):
    '''
    extracts the text part of the json file and returns as a numpy array
    and sets it to lower case
    '''
    data = []
    with open(file_name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            data.append(json.loads(line)['text'].lower())  ##TODO : Check if lower needed
        data = np.array(data)
    return data


def remove_punctuation(texts):
    '''
    removes the punctuations
    '''
    stripPunct = str.maketrans('', '', string.punctuation)
    new = np.array([i.translate(stripPunct) for i in texts])
    return new


def cleanup_data(data):
    """
    contains lambda to remove punctuation
    add additional cleaning functions over here if needed in future
    """
    f = lambda x: remove_punctuation(x)
    data = f(data)
    return data


# @timeit
def generate_N_grams(data, ngram=5):
    """
    Takes and input a np arrary of texts.
    Breaks it into list of 5-grams inside a Dataframe

    # label meanings:
    # 0: no error in middle word
    # 1: With error in middle word

    """

    new_dataset = []

    for n, text in tqdm(enumerate(data)):
        # TODO https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/#:~:text=N%2Dgrams%20are%20continuous%20sequences,(Natural%20Language%20Processing)%20tasks.

        r = r'\S*\d+\S*'  # Remove alpha-num words ; https://stackoverflow.com/a/65105960/5959601
        text = re.sub(r, '', text)
        text = text.split()
        text[:] = [tup for tup in text if tup.isalpha()]
        text[:] = [tup for tup in text if tup.isascii()]

        for i in range(0, len(text) - ngram + 1):
            x = []
            for j in range(5):
                x.append(text[i + j])
            new_dataset.append([x])

    labels = [0] * len(new_dataset)
    return new_dataset, labels  # new_dataset: ndarray(13499,1,5) ; labels : ndarray(13499)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.words = data[0]
        self.labels = data[1]

    def __getitem__(self, i):
        # x = self.words
        word = self.words[i][0]
        label = int(self.labels[i])
        return (word, label)

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    temp_x, temp_y = [], []
    # print(batch, type(batch), len(batch))
    for x, y in batch:
        temp_x.append(x)
        temp_y.append(y)
    return temp_x, temp_y


def convert_to_pytorch_dataset(train_data, val_data, args):
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn,
                                  # num_workers=1,pin_memory=True
                                  )

    val_dataset = MyDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader


def initialize_model(args, device):
    input_dim = alph_len * 3
    hidden_dim = args.hidden_dim  # TODO : Iterate over different hidden dim sizes
    layer_dim = args.hidden_layers
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model = model.to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, criterion, optimizer


def train_model(train_loader, model, criterion, optim, writer, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    correct = 0
    # model.train()
    for i, data in enumerate(tqdm(train_loader)):
        X_vec, Y_vec, X_token = vectorize_data2(data)
        X_vec = X_vec.to(device)
        # X_vec = torch.unsqueeze(X_vec, 1).requires_grad_()  # (n_words,228) --> (n_words , 1, 228)
        Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)
        optim.zero_grad()

        outputs = model(X_vec)  # (n_words, 2)#
        loss = criterion(outputs, Y_vec)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == Y_vec).sum()
        # c = collections.Counter(Y_vec.cpu().detach().numpy())
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
    to_print = np.empty((1, 7))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, X_token = vectorize_data2(data)  # xx shape:
            X_vec = X_vec.to(device)
            # X_vec = torch.unsqueeze(X_vec, 1).requires_grad_()  # (n_words,228) --> (n_words , 1, 228)
            Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)

            outputs = model(X_vec)  # (n_words, 2)

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

            temp_to_print = np.column_stack((X_token, Y_vec.cpu(), predicted.cpu()))
            to_print = np.row_stack((to_print, temp_to_print))

        to_print = pd.DataFrame(to_print)
        to_print.to_csv(os.path.join(args.model_folder, "data2.csv"))
        # to_print.to_csv('data2.csv')
        # mean_val_loss = total_loss / total
        alpha = (len(val_loader.dataset)) / batch_size
        # alpha = 1000 / batch_size
        mean_val_loss = total_loss / alpha
        mean_val_accuracy = 100 * correct / total
        scalar_dict = {'Loss/val': mean_val_loss, 'Accuracy/val': mean_val_accuracy, 'F1_score/f1':f1}
        print(f"mean_val_loss:{mean_val_loss} mean_val_acc:{mean_val_accuracy} , f1_score={f1},total_correct={correct},"
              f"total_samples={total}")
    # accuracy = 100 * correct / total
    # print(f" Word = {X_token[600]} Prediction= {predicted[600]} loss = {loss.item()} accuracy= {accuracy} f1_Score={f1}")
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy.cpu(), f1


def binarize(tokens, alph):
    words = tokens[:-1]
    label = tokens[-1]

    bin = []

    for token in words:
        bin_beg = [0] * len(alph)
        bin_middle = [0] * len(alph)
        bin_end = [0] * len(alph)

        bin_beg[alph.index(token[0])] += 1
        bin_end[alph.index(token[-1])] += 1

        for i in range(1, len(token) - 1):
            bin_middle[alph.index(token[i])] += 1

        bin_all = bin_beg + bin_middle + bin_end
        bin.append(bin_all)

    return torch.tensor(bin), torch.tensor(int(float(label))), words


def vectorize_data(data_arr):
    # https://arxiv.org/pdf/1608.02214.pdf
    '''
    :param data_arr: tuple(2) ; (list[200],list[200])
    :return:
    '''
    data_arr = np.column_stack((data_arr[0], data_arr[1]))
    data_arr = insert_errors(data_arr)  # (batch_size *6)
    # X_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, len(alph) * 3))
    X_vec = torch.zeros((len(data_arr), 5, len(alph) * 3))  # (batch_len * 5 * 228 )
    Y_vec = torch.zeros((len(data_arr), 1))
    X_token = []
    # TODO:
    # make X_token as np.array so that it can contain strings
    # shuflle it same way as for tensor ... X_token = X_token[r]

    for m, mini_batch_tokens in enumerate(zip(*[iter(data_arr)])):
        # mini_batch_tokens : tuple(1) ; (ndarray[6])

        X_token_m = []
        x_mini_batch = torch.zeros((5, len(alph) * 3))  # (1,228)
        y_mini_batch = torch.zeros((1, 1))  # (1,1)

        for j, token in enumerate(mini_batch_tokens):
            x, y, z = binarize(token, alph)
            x_mini_batch, y_mini_batch, x_token = x, y, z
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)
        sys.stdout.flush()

    r = torch.randperm(X_vec.size()[0])
    X_vec = X_vec[r]
    Y_vec = Y_vec[r]

    X_token = np.asarray(X_token)
    X_token = X_token[r]
    return X_vec, Y_vec, X_token


def binarize2(tokens, isLabelVector=False):
    bin = []

    if isLabelVector:
        return torch.tensor(int(float(tokens)))
    bin_beg = [0] * len(alph)
    bin_middle = [0] * len(alph)
    bin_end = [0] * len(alph)

    bin_beg[alph.index(tokens[0])] += 1
    bin_end[alph.index(tokens[-1])] += 1

    for i in range(1, len(tokens) - 1):
        bin_middle[alph.index(tokens[i])] += 1

    bin_all = bin_beg + bin_middle + bin_end
    bin.append(bin_all)
    return torch.tensor(bin)


def vectorize_data2(data_arr):
    '''
    Uses np broadcasting instead of earlier technique
    :param data_arr: ndarray (batch_len,6) ; e.g. [[['big' 'brother' 'ninetepn' 'eightyfour' 'big' '0']]]
    '''
    data_arr = np.column_stack((data_arr[0], data_arr[1]))
    data_arr = insert_errors(data_arr)
    # X_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, len(alph) * 3))
    X_vec = torch.zeros((len(data_arr), 5, len(alph) * 3))  # (batch_len * 5 * 228 )
    Y_vec = torch.zeros((len(data_arr), 1))
    X_token = data_arr[:, :5]

    func3 = np.frompyfunc(binarize2, 2, 1)
    a = data_arr[:, :5]
    b = data_arr[:, -1]
    X = func3(a, False)
    Y = func3(b, True)

    for i, x in enumerate(X):
        for j, y in enumerate(x):
            X_vec[i][j] = y

    for i, y in enumerate(Y):
        Y_vec[i] = y

    # X_vec = torch.from_numpy(X_vec)
    # Y_vec = torch.from_numpy(Y)
    r = torch.randperm(X_vec.size()[0])
    X_vec = X_vec[r]
    Y_vec = Y_vec[r]

    # X_token = np.asarray(X_token)
    X_token = X_token[r]

    return X_vec, Y_vec, X_token


# @timeit
def insert_errors(data):  #
    '''

    :param data: ndarray (batch_size,2)
    :return: data : ndarray ( ?? ,2)
    '''
    # print('data shape before ', np.shape(data))
    temp = []
    for i, x in enumerate(data[:, 2]):
        switch_val = get_rand123()
        if switch_val == 1:
            if get_rand01() == 1:
                # Type 1: Replace a character
                yy = np.array2string(x).replace("'", "")
                rep_char = int2char(np.random.randint(0, 26))
                rep_pos = np.random.randint(low=0, high=len(yy))
                # false_word = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
                false_str = data[i][:-1].copy()
                false_str[2] = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
                temp.append(false_str)
                # [i][2] = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
        elif switch_val == 2:

            if get_rand01() == 1 and len(x) > 1:
                # Type 2: delete a character
                yy = np.array2string(x).replace("'", "")
                rep_pos = np.random.randint(low=0, high=len(yy))
                # temp.append(yy[0:rep_pos] + yy[rep_pos + 1:])
                false_str = data[i][:-1]
                false_str[2] = yy[0:rep_pos] + yy[rep_pos + 1:]
                temp.append(false_str)
        elif switch_val == 3:
            pass
    x2 = np.ones((len(temp)))
    x = np.column_stack((temp, x2))
    data = np.concatenate((data, x))
    # print('data shape after ', np.shape(data))
    return data


def main(args, device):
    writer = SummaryWriter()
    logger = get_logger(args.output_folder, args.exp_name)
    train_data = get_wikipedia_text(os.path.join(args.data_folder, args.input_file))
    train_data = cleanup_data(train_data)
    train_data = generate_N_grams(train_data)
    val_data = get_wikipedia_text(os.path.join(args.data_folder, args.val_file))
    val_data = cleanup_data(val_data)
    val_data = generate_N_grams(val_data)
    # dataz = np.load('data\\5_gram_dataset.npz')
    # dataz = np.load(os.path.join(args.data_folder, args.input_file))
    # data = (dataz['arr_0'], dataz['arr_1'])
    train_loader, val_loader = convert_to_pytorch_dataset(train_data, val_data, args)
    model, criterion, optim = initialize_model(args, device)
    # model = nn.DataParallel(model)

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
    print("LSTM Spelling Classifier with Context")
    print(vars(args))
    print()
    main(args, device)
    print(datetime.now() - start)
