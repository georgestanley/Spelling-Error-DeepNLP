# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import re

import numpy as np
import torch
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd

import lstm_spell_classifier_w_context, lstm_spell_classifier_w_context_onehot

app = Dash(__name__)


def convert_to_numpy_valdata(words):
    non_ascii_keys = []
    for x in words.keys():
        if x.isascii() != True:
            non_ascii_keys.append(x)
    for x in non_ascii_keys:
        del words[x]

    x1 = np.array(list(words.keys()))
    # x2 = np.zeros(x1.size)
    x2 = np.array(list(words.values()))
    return (x1, x2)

def generate_N_grams_onehot(data):
    new_dataset = []
    text = "* * " + data[0][0] + " * *"
    ngram = 5

    r = r'\S*\d+\S*'
    text = re.sub(r, '', text)
    text = text.split()
    # text[:] = [tup for tup in text if tup.isalpha()]
    # text[:] = [tup for tup in text if tup.isascii()]

    for i in range(0, len(text) - ngram + 1):
        x = []
        for j in range(5):
            x.append(text[i + j])
        new_dataset.append(' '.join(x))

    new_dataset = np.array(new_dataset)
    labels = [0] * len(new_dataset)
    labels = np.array(labels)

    return new_dataset, labels


def generate_N_grams(data):
    new_dataset = []
    text = "* * " + data[0][0] + " * *"
    ngram = 5

    r = r'\S*\d+\S*'
    text = re.sub(r, '', text)
    text = text.split()
    # text[:] = [tup for tup in text if tup.isalpha()]
    # text[:] = [tup for tup in text if tup.isascii()]

    for i in range(0, len(text) - ngram + 1):
        x = []
        for j in range(5):
            x.append(text[i + j])
        new_dataset.append([x])

    labels = [0] * len(new_dataset)

    return new_dataset, labels


def test_model_lstm_context_semi_character(data):
    PATH = "results//lstm_context//lr0.001_bs512_optimAdam_hidden_dim512_hidden_layers2_//20220803122815_models//ckpt_best_43.pth"

    device = 'cuda'
    model, criterion, _ = lstm_spell_classifier_w_context.initialize_model(hidden_dim=512, hidden_layers=2, lr=0.001,
                                                                           device=device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    val_data = {data: 0}
    val_data = convert_to_numpy_valdata(val_data)
    val_data = generate_N_grams(val_data)
    _, val_loader = lstm_spell_classifier_w_context.convert_to_pytorch_dataset(val_data, val_data, batch_size=1)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, X_token = lstm_spell_classifier_w_context.vectorize_data2(data, with_error=False,
                                                                                    shuffle=False)  # xx shape:
            X_vec = X_vec.to(device)
            outputs = model(X_vec)  # (n_words, 2)
            _, predicted = torch.max(outputs.data, 1)
            print(outputs.data, predicted, X_token)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            error_grams = X_token[indexes]
            error_words = [x[2] for x in error_grams]

            return error_words

def test_model_lstm_context_one_hot(data):
    PATH = "results//lstm_context_onehot//lr0.001_bs512_optimAdam_hidden_dim512_hidden_layers2_" \
           "//20220721103824_models//ckpt_best_37.pth"

    device = 'cuda'
    model, criterion, _ = lstm_spell_classifier_w_context_onehot.initialize_model(hidden_dim=512, hidden_layers=2, lr=0.001,
                                                                           device=device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    val_data = {data: 0}
    val_data = convert_to_numpy_valdata(val_data)
    val_data = generate_N_grams_onehot(val_data)
    _, val_loader = lstm_spell_classifier_w_context_onehot.convert_to_pytorch_dataset(val_data, val_data, batch_size=1000)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, sentence_length = lstm_spell_classifier_w_context_onehot.one_hot_encode_data(data=list(data[0]),
                                                                                               with_error=False,labels=data[1],
                                                                                               shuffle=False, maxlen=60)  # xx shape:
            X_vec = X_vec.type(torch.FloatTensor).to(device)
            sent_len = torch.tensor(sentence_length, device=device)
            outputs = model(X_vec, sent_len)  # (n_words, 2)

            _, predicted = torch.max(outputs.data, 1)
            print(outputs.data, predicted)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            d = np.array(list(data[0]))
            print(d)
            error_grams = d[indexes]
            error_words = [x.split()[2] for x in error_grams]

            return error_words


def test_model_lstm_wo_context(data):
    PATH = "results//lstm_context_onehot//lr0.001_bs512_optimAdam_hidden_dim512_hidden_layers2_" \
           "//20220721103824_models//ckpt_best_37.pth"

    device = 'cuda'
    model, criterion, _ = lstm_spell_classifier_w_context_onehot.initialize_model(hidden_dim=512, hidden_layers=2,
                                                                                  lr=0.001,
                                                                                  device=device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    val_data = {data: 0}
    val_data = convert_to_numpy_valdata(val_data)
    val_data = generate_N_grams_onehot(val_data)
    _, val_loader = lstm_spell_classifier_w_context_onehot.convert_to_pytorch_dataset(val_data, val_data,
                                                                                      batch_size=1000)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, sentence_length = lstm_spell_classifier_w_context_onehot.one_hot_encode_data(
                data=list(data[0]),
                with_error=False, labels=data[1],
                shuffle=False, maxlen=60)  # xx shape:
            X_vec = X_vec.type(torch.FloatTensor).to(device)
            sent_len = torch.tensor(sentence_length, device=device)
            outputs = model(X_vec, sent_len)  # (n_words, 2)

            _, predicted = torch.max(outputs.data, 1)
            print(outputs.data, predicted)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            d = np.array(list(data[0]))
            print(d)
            error_grams = d[indexes]
            error_words = [x.split()[2] for x in error_grams]

            return error_words




app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Spelling Error Detection
    '''),

    dcc.Input(id='input_text', type='text'),
    html.Button('Evaluate text', id='evaluate', n_clicks=0),
    html.Div(id='container-button-basic1',
             children='Enter a value and press submit'),
    html.Div(id='container-button-basic2',
             children='Enter a value and press submit2')
])


@app.callback(
    [Output('container-button-basic1', 'children'),
    Output('container-button-basic2', 'children')],
    Input('evaluate', 'n_clicks'),
    State('input_text', 'value')
)
def update_output(n_clicks, value):
    results1 = test_model_lstm_context_semi_character(value)
    results2 = test_model_lstm_context_one_hot(value)

    return results1,results2

if __name__ == '__main__':
    app.run_server(debug=False)
