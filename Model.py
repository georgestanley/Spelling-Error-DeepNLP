import torch.nn as nn
import torch.nn.functional as F
import torch

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