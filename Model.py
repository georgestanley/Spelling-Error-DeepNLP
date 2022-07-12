import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)

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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        #print(sequence_lengths)
        out = torch.stack([out[i, length] for i, length in enumerate(sequence_lengths)])  # sequence_lengths is an array containing the length of each sequence without padding
        out = self.fc(out)

        #out = self.fc(out[:, -1, :])
        #out = self.sig(out)
        # out.size() --> 100, 10

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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        #print(sequence_lengths)
        #out = torch.stack([out[i, length] for i, length in enumerate(sequence_lengths)])  # sequence_lengths is an array containing the length of each sequence without padding
        #out = self.fc(out)

        out = self.fc(out[:, -1, :])
        #out = self.sig(out)
        # out.size() --> 100, 10

        return out
