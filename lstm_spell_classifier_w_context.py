import pandas as pd
import string, argparse, json, os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from Model import MLPNetwork, RNN, LSTMModel
import sys, random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.utils import get_rand01, check_dir, int2char, get_logger
# import wandb
from sklearn.metrics import f1_score
from datetime import datetime
from nltk.corpus import stopwords

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
    parser.add_argument('--hidden_dim',type=int, default=100,help='LSTM hidden layer Dim')
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

        data = pd.DataFrame(data, columns=['text'])

        # data = f.read()
        # words = json.loads(data)
    return data


def remove_punctuation(text):
    '''

    :param text: String
    :return: ans: String
    '''
    ans = ""
    for i in text:
        if i not in string.punctuation:
            ans += i
    return ans


def cleanup_data(data):
    """
    :param: data :Pandas dataframe [1 column]
    :returns data : Pandas Dataframe [1 column]
    """
    data['text'] = data['text'].apply(lambda x: remove_punctuation(x))
    return data


def generate_N_grams(data, ngram=5):
    """
    Takes and input a Dataframe of texts.Breaks it into list of 5-grams inside a Dataframe
    :param data: Pandas dataframe [1 Column]
    :param ngram: int
    :return: new_dataset: Pandas dataframe
    """

    new_dataset = []

    for n, text in data.iterrows():
        # TODO https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/#:~:text=N%2Dgrams%20are%20continuous%20sequences,(Natural%20Language%20Processing)%20tasks.
        text = text.values[0].split()

        for i in range(0, len(text) - ngram + 1):
            x = []
            for j in range(5):
                # print(f"{i},{j},{text[i + j]}")
                x.append(text[i + j])
            new_dataset.append([x])

    new_dataset = pd.DataFrame(new_dataset, columns=['inputs'])
    new_dataset['labels'] = 0
    # label meanings:
    # 0: no error in middle word
    # 1: With error in middle word

    return new_dataset


def convert_to_numpy(data):
    '''
    Ignore for now.
    :param data:
    :return: data
    '''
    return data


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.words = data['inputs']
        self.labels = data['labels']

        # self.words = words
        # self.labels = labels

    def __getitem__(self, i):
        word = self.words[i]
        label = int(self.labels[i])
        return (word, label)

    def __len__(self):
        return len(self.labels)


def convert_to_pytorch_dataset(data):
    train_dataset = MyDataset(data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = MyDataset(data)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    return train_dataloader, val_dataloader


def initialize_model(n_hidden_layers=1):
    input_dim = 228
    hidden_dim = args.hidden_dim  # TODO : Iterate over different hidden dim sizes
    layer_dim = n_hidden_layers
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, criterion, optimizer


def main(args):
    os.environ["WANDB_MODE"] = "dryrun"
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
    data = get_wikipedia_text(os.path.join(args.data_folder, "dev_10.jsonl"))
    data = cleanup_data(data)
    data = generate_N_grams(data)
    data = convert_to_numpy(data)
    train_loader, val_loader = convert_to_pytorch_dataset(data)
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
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.plot(np.arange(n_epoch + 1), train_losses)
    ax1.set_title('Train Loss')
    ax2.plot(np.arange(n_epoch + 1), val_losses)
    ax2.set_title('Val Loss')
    ax3.plot(np.arange(n_epoch + 1), val_accuracies)
    ax3.set_title('Val Acc')
    ax4.plot(np.arange(n_epoch + 1), val_f1s)
    ax4.set_title('Val F1')

    plt.savefig(fname=os.path.join(args.model_folder, "plot.png"))
    plt.show()

    return


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cpu")
    print("LSTM Spelling Classifier")
    print(vars(args))
    print()
    main(args)
