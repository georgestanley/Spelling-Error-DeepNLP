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
#source:https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

#observations:
#the network starts outputing 0 towards later stages.
import random

import numpy

from nltk.corpus.reader import WordListCorpusReader
import numpy as np
from nltk.corpus import stopwords
from utils import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_iters = 100000
print_every = 1000
plot_every = 1000


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
    words = [word for word in words if word not in stopwords_list]
    # print(xx)

    # print(words)
    data_arr = np.array((words, np.ones(len(words),dtype=numpy.int))).T
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
            # rep_char = int2char(np.random.randint(0, 26))
            rep_pos = np.random.randint(low=0, high=len(yy))
            # yy[rep_pos]=rep_char
            # print("rep_char=",rep_char,"rep_pos=",rep_pos)
            x_temp.append(yy[0:rep_pos] + yy[rep_pos + 1:])
            # print("new word=",yy)
            # x_temp.append(y)

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

    x_temp_arr = np.array((x_temp, np.zeros(len(x_temp),dtype=numpy.int ))).T
    print("Shape of Error data", np.shape(x_temp_arr))
    data_arr = np.concatenate((data_arr, x_temp_arr))
    print("Shape after adding new data", np.shape(data_arr))
    # print(data_arr)

    return data_arr

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# Create an MLP network
class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(57, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create a CNN network
# compare performance of both

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        #self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h = nn.LSTM(input_size,hidden_size)
        #self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.LSTM(input_size,hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input,hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(model, criterion, input,result, learning_rate):
    hidden = model.initHidden()
    model.zero_grad()
    output = None

    for i in range(input.size()[0]):
        output, hidden = model(input[i],hidden)

    loss = criterion(output, result)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def randomTrainingExample(data_arr):
    index = np.random.choice(data_arr.shape[0])
    #word = data_arr[index][0]
    #result = data_arr[index][1]
    word = lineToTensor(data_arr[index][0])
    result = torch.tensor([int(data_arr[index][1])], dtype=torch.long)
    return word,result

    pass


def randomIndex(max_len):
    return random.randint(0,max_len-1)


def classFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

def main():
    n_letters = len(all_letters)
    n_classes = 2
    data_arr = preapre_dataset()
    data_arr = insert_errors(data_arr)
    print(data_arr.shape)

    n_hidden = 128
    #model = MLPNetwork()
    model = RNN(n_letters,n_hidden,n_classes)
    #model2 = nn.LSTM(input_size=n_letters,hidden_size=n_hidden)
    criterion = nn.NLLLoss()

    word = lineToTensor(data_arr[0][0])
    #op = model2(word,torch.zeros(1,n_hidden))
    output, hidden = model(word[0],torch.zeros(1, n_hidden))
    result = torch.tensor([ int(data_arr[0][1])], dtype=torch.long)
    #l = criterion(output,result)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    for iter in range(1, n_iters + 1):
        #if iter%1000==0:
            #print('Iter:',iter)
            #pass
        index = randomIndex(data_arr.shape[0])
        if len(data_arr[index][0])==0:
            continue
        #word,result = randomTrainingExample(data_arr)
        #output, loss = train(model, criterion, category_tensor, line_tensor, learning_rate=0.005)
        actual_word=data_arr[index]
        word = lineToTensor(data_arr[index][0])
        result = torch.tensor([int(data_arr[index][1])], dtype=torch.long)
        output, loss = train(model, criterion, word,result, learning_rate=0.005)

        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
           guess = classFromOutput(output)
           correct = '✓' if guess ==result  else '✗ (%s)' % actual_word[0]
           print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, 'xxx', loss, actual_word, guess, correct))
        #
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


    plt.figure()
    print(all_losses)
    plt.plot(all_losses)


if __name__ == "__main__":
    main()