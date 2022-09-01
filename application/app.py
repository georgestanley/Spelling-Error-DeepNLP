# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import os

import argparse

import re

import numpy as np
import torch
from dash import Dash, html, dcc, Input, Output, State

import application.lstm_spell_classifier_w_context as lstm_spell_classifier_w_context, \
    application.lstm_spell_classifier_w_context_onehot as lstm_spell_classifier_w_context_onehot, \
    application.lstm_spell_classifier_wo_context as lstm_spell_classifier_wo_context

app = Dash(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="console OR webapp")
    parser.add_argument('--model1_path', type=str,
                        default='data//trained_models//semi_character_w_context.pth',
                        help='trained Model 1 ==> Semi-Character-Encoding / With Context')
    parser.add_argument('--model2_path', type=str, default='data//trained_models//onehot_w_context.pth',
                        help='trained Model 2 ==> One-hot-Encoding / With Context')
    parser.add_argument('--model3_path', type=str,
                        default='data//trained_models//semi_character_wo_context.pth',
                        help='trained Model 3 ==> Semi-Character-Encoding / Without Context')
    args = parser.parse_args()

    return args


def initialize_models():
    device = 'cuda'
    ###### LOAD THE MODELS ######

    PATH_SEMI_CHARACTER = "D://Freiburg//MasterProject//results//lstm_context" \
                          "//lr0.001_bs512_optimAdam_hidden_dim512_hidden_layers2_//20220803122815_models//ckpt_best_43.pth"
    PATH_SEMI_CHARACTER = args.model1_path
    model_semi_character_w_context, _, _ = lstm_spell_classifier_w_context.initialize_model(hidden_dim=512,
                                                                                            hidden_layers=2,
                                                                                            lr=0.001,
                                                                                            device=device)
    model_semi_character_w_context.load_state_dict(torch.load(PATH_SEMI_CHARACTER))
    model_semi_character_w_context.eval()

    ######
    PATH_ONE_HOT = "results//lstm_context_onehot//lr0.001_bs512_optimAdam_hidden_dim512_hidden_layers2_" \
                   "//20220721103824_models//ckpt_best_37.pth"

    PATH_ONE_HOT = args.model2_path
    model_one_hot, _, _ = lstm_spell_classifier_w_context_onehot.initialize_model(hidden_dim=512, hidden_layers=2,
                                                                                  lr=0.001,
                                                                                  device=device)
    model_one_hot.load_state_dict(torch.load(PATH_ONE_HOT))
    model_one_hot.eval()

    ######
    PATH_WO_CONTEXT = "results//lstm_noncontext//lr0.01_bs1024_optimAdam_hidden_dim1024_hidden_layers2_//20220802190816_models" \
                      "//ckpt_best_47.pth "
    PATH_WO_CONTEXT = args.model3_path
    model_semi_character_wo_context, criterion, _ = lstm_spell_classifier_wo_context.initialize_model(n_hidden_layers=2,
                                                                                                      hidden_dim=1024,
                                                                                                      lr=0.01,
                                                                                                      device=device)
    model_semi_character_wo_context.load_state_dict(torch.load(PATH_WO_CONTEXT))
    model_semi_character_wo_context.eval()

    print('All Models Imported')
    return device, model_semi_character_w_context, model_one_hot, model_semi_character_wo_context


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


def evaluate_lstm_context_semi_character(model, data, device='cuda'):
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
            # print(outputs.data, predicted, X_token)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            error_grams = X_token[indexes]
            error_words = [x[2] for x in error_grams]
            # print("Error Words = ", error_words)
            return error_words


def evaluate_lstm_context_one_hot(model, data, device='cuda'):
    val_data = {data: 0}
    val_data = convert_to_numpy_valdata(val_data)
    val_data = generate_N_grams_onehot(val_data)
    _, val_loader = lstm_spell_classifier_w_context_onehot.convert_to_pytorch_dataset(val_data, val_data,
                                                                                      batch_size=1000,
                                                                                      val_shuffle=False)
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
            # print(outputs.data, predicted)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            d = np.array(list(data[0]))
            # print(d)
            error_grams = d[indexes]
            error_words = [x.split()[2] for x in error_grams]
            # print("Error Words = ", error_words)
            return error_words


def evaluate_lstm_wo_context(model, data, device='cuda'):
    # val_data = {data: 0}
    val_data = dict.fromkeys(data.split(), 0)
    val_data = lstm_spell_classifier_wo_context.convert_to_numpy_valdata(val_data)
    _, val_loader = lstm_spell_classifier_wo_context.convert_to_pytorch_dataset(val_data, val_data,
                                                                                batch_size=1000)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X_vec, Y_vec, X_token = lstm_spell_classifier_wo_context.vectorize_data(data, with_error=False,
                                                                                    shuffle=False)  # xx shape:
            X_vec = torch.unsqueeze(X_vec, 1).requires_grad_().to(device)  # (n_words,228) --> (n_words , 1, 228)
            outputs = model(X_vec)  # (n_words, 2)

            _, predicted = torch.max(outputs.data, 1)
            # print(outputs.data, predicted, X_token)
            l = predicted.tolist()
            indexes = [i for i, x in enumerate(l) if x == 1]
            error_grams = X_token[indexes]
            error_words = [x[0] for x in error_grams]
            # print("Error Words = ", error_words)
            return error_words


app.layout = html.Div(children=[
    html.H1(children='Spelling Error Detection', style={'textAlign': 'center'}),
    html.H3(children='Master Project by Stanley George', style={'textAlign': 'center'}),

    dcc.Input(id='input_text', type='text', style={'width': '70%'}),
    html.Button('Evaluate text', id='evaluate', n_clicks=0, style={'width': '20%'}),
    html.Div(id='div1', children=[html.Div(id='div1a', children='Model 1:', style={'display': 'inline-block'}),
                                  html.Div(id='container-button-basic1', children='',
                                           style={'display': 'inline-block'})]),
    html.Div(id='div2', children=[html.Div(id='div2a', children='Model 2:', style={'display': 'inline-block'}),
                                  html.Div(id='container-button-basic2', children='',
                                           style={'display': 'inline-block'})]),
    html.Div(id='div3', children=[html.Div(id='div3a', children='Model 3:', style={'display': 'inline-block'}),
                                  html.Div(id='container-button-basic3', children='',
                                           style={'display': 'inline-block'})]),

])


@app.callback(
    [Output('container-button-basic1', 'children'),
     Output('container-button-basic2', 'children'),
     Output('container-button-basic3', 'children')],
    Input('evaluate', 'n_clicks'),
    State('input_text', 'value')
)
def update_output(n_clicks, value):
    results1 = evaluate_lstm_context_semi_character(model=model_semi_character_w_context, data=value)
    results2 = evaluate_lstm_context_one_hot(model=model_one_hot, data=value)
    results3 = evaluate_lstm_wo_context(model=model_semi_character_wo_context, data=value)

    return ','.join(results1), ','.join(results2), ','.join(results3)


def main():
    print('Model 1 ==>(Semi-Character-Encoding / With Context)')
    print('Model 2 ==>(One-hot-Encoding / With Context)')
    print('Model 3 ==>(Semi-Character-Encoding / Without Context)')
    while True:
        print('')
        value = input('Sentence to evaluate:')
        if value.strip() == '':
            continue
        results1 = evaluate_lstm_context_semi_character(model=model_semi_character_w_context, data=value)
        results2 = evaluate_lstm_context_one_hot(model=model_one_hot, data=value)
        results3 = evaluate_lstm_wo_context(model=model_semi_character_wo_context, data=value)
        print('Model 1 :', results1)
        print('Model 2 :', results2)
        print('Model 3 :', results3)
    pass


def extract_data_from_file(file_path):
    with open(file_path) as f:
        lines = f.read()
        # print(lines)
        return lines
    pass


def file_eval():
    print('Model 1 ==>(Semi-Character-Encoding / With Context)')
    print('Model 2 ==>(One-hot-Encoding / With Context)')
    print('Model 3 ==>(Semi-Character-Encoding / Without Context)')
    while True:
        file_path = input('Please provide the file path. Kindly provide on .txt files:')
        if (os.path.exists(file_path)):
            value = extract_data_from_file(file_path)
            results1 = evaluate_lstm_context_semi_character(model=model_semi_character_w_context, data=value)
            results2 = evaluate_lstm_context_one_hot(model=model_one_hot, data=value)
            results3 = evaluate_lstm_wo_context(model=model_semi_character_wo_context, data=value)
            print('Model 1 :', results1)
            print('Model 2 :', results2)
            print('Model 3 :', results3)
        else:
            print('Please provide valid file path')

    pass


if __name__ == '__main__':
    print('Initializing ...')
    args = parse_arguments()
    device, model_semi_character_w_context, model_one_hot, model_semi_character_wo_context = initialize_models()
    if args.mode == 'console':
        main()
    elif args.mode == 'webapp':
        app.run_server(debug=False)
    elif args.mode == 'file_eval':
        file_eval()
    else:
        print('Please select between \'console\' OR \'webapp\' OR \'file_eval\' ')

    # test_model_lstm_context_semi_character('We need something concrete for the world')
