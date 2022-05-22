import string, argparse, json, os, re
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from Model import LSTMModel
import sys
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.utils import get_rand01, check_dir, int2char, get_logger
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
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
alph_len = len(alph)

exp_id = datetime.now().strftime('%Y%m%d%H%M%S')


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--input_file', type=str, default='dev_10.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=1000, help='batch_size')
    parser.add_argument('--optim', type=str, default="Adam", help="optimizer to use")
    parser.add_argument('--hidden_dim', type=int, default=100, help='LSTM hidden layer Dim')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()
    hparam_keys = ["lr", "bs", "optim"]  # changed from loss to size
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'lstm_context', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "{}_models".format(exp_id)))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


#@timeit
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

        # data = pd.DataFrame(data, columns=['text'])
        data = np.array(data)

        # data = f.read()
        # words = json.loads(data)
    return data


#@timeit
def remove_punctuation(texts):
    '''

    :param text: String
    :return: ans: String
    '''
    ans = ""
    stripPunct = str.maketrans('', '', string.punctuation)
    new = np.array([i.translate(stripPunct) for i in texts])
    return new


#@timeit
def cleanup_data(data):
    """
    :param: data :Pandas dataframe [1 column]
    :returns data : Pandas Dataframe [1 column]
    """
    # data['text'] = data['text'].apply(lambda x: remove_punctuation(x))
    f = lambda x: remove_punctuation(x)
    data = f(data)
    return data


#@timeit
def generate_N_grams(data, ngram=5):
    """
    Takes and input a Dataframe of texts.Breaks it into list of 5-grams inside a Dataframe
    :param data: Pandas dataframe [1 Column]
    :param ngram: int
    :return: new_dataset: Pandas dataframe
    """

    new_dataset = []

    for n, text in tqdm(enumerate(data)):
        # TODO https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/#:~:text=N%2Dgrams%20are%20continuous%20sequences,(Natural%20Language%20Processing)%20tasks.

        r = r'\S*\d+\S*'  # Remove alpha-num words ; https://stackoverflow.com/a/65105960/5959601
        text = re.sub(r, '', text)
        text = text.split()
        text[:] = [tup for tup in text if  tup.isalpha()]
        text[:] = [tup for tup in text if  tup.isascii()]

        # if 'sédar' in text:
        #     print('Seadr found')
        # for x in text:
        #     if not x.isalpha():
        #         text.remove(x)
        #
        # for x in text:
        #     if not x.isascii():
        #         print('Removed',x)
        #         text.remove(x)

        for i in range(0, len(text) - ngram + 1):
            x = []
            for j in range(5):
                # print(f"{i},{j},{text[i + j]}")
                x.append(text[i + j])
            new_dataset.append([x])

    # new_dataset = pd.DataFrame(new_dataset, columns=['inputs'])
    # new_dataset['labels'] = 0

    new_dataset = np.array(new_dataset)
    labels = np.zeros(len(new_dataset))

    # label meanings:
    # 0: no error in middle word
    # 1: With error in middle word

    return new_dataset, labels


def convert_to_numpy(data):
    '''
    Ignore for now.
    :param data:
    :return: data
    '''
    return data


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.words = data[0]
        self.labels = data[1]

        # self.words = words
        # self.labels = labels

    def __getitem__(self, i):
        # x = self.words
        word = self.words[i][0]
        label = int(self.labels[i])
        return (word, label)

    def __len__(self):
        return len(self.labels)


#@timeit
def collate_fn(batch):
    temp_x, temp_y = [], []
    # print(batch)
    # print(type(batch))
    # print(len(batch))
    for x, y in batch:
        temp_x.append(x)
        temp_y.append(y)

    return temp_x, temp_y


#@timeit
def convert_to_pytorch_dataset(data):
    train_dataset = MyDataset(data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    val_dataset = MyDataset(data)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader

#@timeit
def initialize_model(n_hidden_layers=1):
    input_dim = alph_len*3
    hidden_dim = args.hidden_dim  # TODO : Iterate over different hidden dim sizes
    layer_dim = n_hidden_layers
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, criterion, optimizer

#@timeit
def train_model(train_loader, model, criterion, optim, epoch):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        X_vec, Y_vec, X_token = vectorize_data2(data)  # xx shape:
        X_vec = X_vec.to(device)
        # X_vec = torch.unsqueeze(X_vec, 1).requires_grad_()  # (n_words,228) --> (n_words , 1, 228)
        Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)
        optim.zero_grad()
        outputs = model(X_vec)  # (n_words, 2)#
        loss = criterion(outputs, Y_vec)
        # wandb.log({"train_loss":loss})
        # wandb.watch(model)
        loss.backward()
        optim.step()

        running_loss += loss.item()

    return running_loss

#@timeit
def val_model(val_loader, model, criterion, logger, epoch=0, ):
    # TODO: Improve this validation section
    correct = 0
    total = 0
    f1 = 0

    for i, data in enumerate(val_loader):
        X_vec, Y_vec, X_token = vectorize_data(data)  # xx shape:
        X_vec = X_vec.to(device)
        # X_vec = torch.unsqueeze(X_vec, 1).requires_grad_()  # (n_words,228) --> (n_words , 1, 228)
        Y_vec = torch.squeeze(Y_vec).type(torch.LongTensor).to(device)

        outputs = model(X_vec)  # (n_words, 2)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += Y_vec.size(0)

        loss = criterion(outputs, Y_vec)
        # Total correct predictions
        correct += (predicted == Y_vec).sum()

        f1 = f1_score(predicted.cpu(), Y_vec.cpu())
        # check for an index
        # print(f" Word = {X_token[60]} Prediction= {predicted[60]}")

        break

    accuracy = 100 * correct / total

    print(
        f" Word = {X_token[600]} Prediction= {predicted[600]} loss = {loss.item()} accuracy= {accuracy} f1_Score={f1}")
    # wandb.log({"val_loss": loss.item()})
    # wandb.log({"val_accuracy":accuracy})
    # wandb.log({"f1_score":f1})

    return loss.item(), accuracy.item(), f1


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
    data_arr = insert_errors(data_arr) # (batch_size *6)
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
    data_arr = np.column_stack((data_arr[0], data_arr[1]))
    data_arr = insert_errors(data_arr)
    # X_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, len(alph) * 3))
    X_vec = torch.zeros((len(data_arr), 5, len(alph) * 3))  # (batch_len * 5 * 228 )
    Y_vec = torch.zeros((len(data_arr), 1))
    X_token = data_arr[:,:4]

    func3 = np.frompyfunc(binarize2, 2, 1)
    a= data_arr[:,:4]
    b = data_arr[:,-1]
    X = func3(a,False )
    Y = func3(b,True)

    for i,x in enumerate(X):
        for j, y in enumerate(x):
            X_vec[i][j] = y

    for i,y in enumerate(Y):
            Y_vec[i] = y

    # X_vec = torch.from_numpy(X_vec)
    # Y_vec = torch.from_numpy(Y)

    r = torch.randperm(X_vec.size()[0])
    X_vec = X_vec[r]
    Y_vec = Y_vec[r]

    # X_token = np.asarray(X_token)
    X_token = X_token[r]

    return X_vec, Y_vec, X_token


#@timeit
def insert_errors(data):  #
    '''

    :param data: ndarray (batch_size,2)
    :return: data : ndarray ( ?? ,2)
    '''
    # print('data shape before ', np.shape(data))
    temp = []
    for i, x in enumerate(data[:, 2]):
        if get_rand01() == 1:
            # Type 1: Replace a character
            yy = np.array2string(x).replace("'", "")
            rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # false_word = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
            false_str = data[i][:-1]
            false_str[2] = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]
            temp.append(false_str)
            # [i][2] = yy[0:rep_pos] + rep_char + yy[rep_pos + 1:]

        if get_rand01() == 1 and len(x) > 1:
            # Type 2: delete a character
            yy = np.array2string(x).replace("'", "")
            rep_pos = np.random.randint(low=0, high=len(yy))
            # temp.append(yy[0:rep_pos] + yy[rep_pos + 1:])
            false_str = data[i][:-1]
            false_str[2] = yy[0:rep_pos] + yy[rep_pos + 1:]
            temp.append(false_str)

    x2 = np.ones((len(temp)))
    x = np.column_stack((temp, x2))
    data = np.concatenate((data, x))
    # print('data shape after ', np.shape(data))
    return data


def test_dataloader(my_dataloader):
    for i, (word, label) in enumerate(my_dataloader):
        # print(x)
        print(i, word, label)
        return




def main(args):
    # os.environ["WANDB_MODE"] = "dryrun"
    # wandb.init(project="my-test-project", entity="georgestanley")
    # wandb.config = {
    #     "learning_rate":args.lr,
    #     "bs":args.bs,
    #     "epochs":30
    # }

    logger = get_logger(args.output_folder, args.exp_name)
    model_type = 'RNN'
    n_letters = len(all_letters)
    n_classes = 2
    #data = get_wikipedia_text(os.path.join(args.data_folder, args.input_file))
    #data = cleanup_data(data)
    #data = generate_N_grams(data)
    #data = convert_to_numpy(data)
    dataz = np.load('data\\5_gram_dataset.npz')
    dataz = np.load(os.path.join(args.data_folder,args.input_file))
    data = (dataz['arr_0'],dataz['arr_1'])
    train_loader, val_loader = convert_to_pytorch_dataset(data)
    model, criterion, optim = initialize_model(n_hidden_layers=1)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])

    print("Dataset size: {} samples".format(len(train_loader.dataset)))  # TODO
    logger.info(expdata)
    logger.info('train_data {}'.format(train_loader.dataset.__len__()))  # TODO
    logger.info('val_data {}'.format(val_loader.dataset.__len__()))  # TODO

    n_epoch = args.epochs

    # test_dataloader(train_loader)
    train_losses, val_losses, val_accuracies, val_f1s = [0.0], [0.0], [0.0], [0.0]
    for epoch in range(n_epoch):

        train_loss = train_model(train_loader, model, criterion, optim, epoch)
        val_loss, val_acc, val_f1 = val_model(val_loader, model, criterion, logger)

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

    # create plot

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
    start = datetime.now()
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device}")
    print("LSTM Spelling Classifier")
    print(vars(args))
    print()
    main(args)
    print(datetime.now()-start)

