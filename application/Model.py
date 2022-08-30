import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
'''
# Create an MLP network
class MLPNetwork(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(MLPNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
'''

'''
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
'''

class LSTMModelForOneHotEncodings(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModelForOneHotEncodings, self).__init__()
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, sequence_lengths):
        # x: Tensor(500,1,228)
        # Initialize hidden state with zeros
        self.lstm.flatten_parameters()  # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506#

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = torch.stack([out[i, length] for i, length in enumerate(sequence_lengths)]).to(self.device) # sequence_lengths is an array containing the length of each sequence without padding
        out = self.fc(out)

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        # x: Tensor(500,1,228)
        # Initialize hidden state with zeros
        self.lstm.flatten_parameters() #https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        return out
