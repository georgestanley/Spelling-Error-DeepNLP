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
from Model import MLPNetwork,RNN
import torch.optim as optim
import sys

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_iters = 100000
print_every = 1000
plot_every = 1000
batchsize=100
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
num_epochs=5


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# Create a CNN network
# compare performance of both

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

def binarize (token,label, alph):

    bin_beg =  [0] * len(alph)
    bin_middle = [0] * len(alph)
    bin_end = [0] * len(alph)

    bin_beg[alph.index(token[0])] += 1
    bin_end[alph.index(token[-1])] += 1

    for i in range(1,len(token)-1):
        bin_middle[alph.index(token[i])] += 1

    bin_all = bin_beg + bin_middle + bin_end
    return torch.Tensor(bin_all) , torch.Tensor([int(label)]),token

def vectorize_data(data_arr):
    # https://arxiv.org/pdf/1608.02214.pdf

    X_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, len(alph) * 3))
    Y_vec = torch.zeros((int(len(data_arr) / batchsize), batchsize, 1))
    X_token = []

    for m, mini_batch_tokens in enumerate(zip(*[iter(data_arr)] * batchsize)):
        X_token_m = []
        x_mini_batch = torch.zeros((batchsize, len(alph) * 3))
        y_mini_batch = torch.zeros((batchsize, 1))


        for j, token in enumerate(mini_batch_tokens):
            x,y,z= binarize(token[0],token[1],alph)
            x_mini_batch[j], y_mini_batch[j], x_token = x,y,z
            '''
            if jumble_type == 'NO':
                x_mini_batch[j], x_token = binarize.noise_char(token, noise_type, alph)
            else:
                x_mini_batch[j], x_token = binarize.jumble_char(token, jumble_type, alph)
            '''

            #bin_label = [0] * len(vocab)
            #bin_label[vocab[token]] = 1
            #y_mini_batch[j] = np.array(bin_label)
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)

        percentage = int(m * 100. / (len(data_arr) / batchsize))
        sys.stdout.write("\r%d %% %s" % (percentage, 'train data'))
        # print(str(percentage) + '%'),
        sys.stdout.flush()

    print(X_vec.shape,Y_vec.shape)
    return X_vec , Y_vec,X_token

def main():
    model_type='MLP'
    n_letters = len(all_letters)
    n_classes = 2
    data_arr = preapre_dataset()
    data_arr = insert_errors(data_arr)
    np.random.shuffle(data_arr)
    print(data_arr.shape)

    n_hidden = 128

    if model_type == 'RNN':

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

    elif model_type=='MLP':

        X_train, y_train , X_token = vectorize_data(data_arr)


        #print(X_train.shape)
        model = MLPNetwork(input_dim=228,output_dim=1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for e in range(num_epochs):

            for i  in range(X_train.shape[0]):
                #print(X_train[i].shape, y_train[i].shape)
                optimizer.zero_grad()
                print(X_train[i],y_train[i])

                outputs = model(X_train[i])
                loss = criterion(outputs,y_train[i])
                loss.backward()
                optimizer.step()
                print(outputs[20],X_token[i][20])
                break











if __name__ == "__main__":
    main()