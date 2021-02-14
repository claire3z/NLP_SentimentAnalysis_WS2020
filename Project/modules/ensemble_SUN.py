''"""
Created by: Claire Z. Sun
Date: 2021.02.03
''"""


import torch

# Ensemble model with one single layer fully connected neurons and softmax
class Ensemble(torch.nn.Module):
    def __init__(self,n):
        super(Ensemble, self).__init__()
        self.fc = torch.nn.Linear(n, 3)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.fc(x)
        out = self.softmax(x)
        return out


# Alternative ensemble architecture (1D convolution + 2 fully connected layers and log_softmax activation)
class Ensemble2(torch.nn.Module):
    def __init__(self,n,h):
        super(Ensemble2, self).__init__()
        self.n = n
        self.conv = torch.nn.Conv1d(n, n, 1) # effectively serves as a selection layer
        self.fc1 = torch.nn.Linear(n, h) # h=hidden_state
        self.fc2 = torch.nn.Linear(h, 3)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = self.fc1(x.view(-1, self.n))
        x = self.fc2(x)
        out = self.softmax(x)
        return out
