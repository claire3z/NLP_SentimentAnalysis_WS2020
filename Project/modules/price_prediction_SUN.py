"Owner: Claire SUN"

''"""
Created by: Claire Z. Sun
Date: 2021.02.08
''"""

### References:
# RNN tutorial:     https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
# LSTM turorial:    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# LSTM tutorial:    https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=_BcDEjcABRVz
# Hidden states:    https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384/2


import numpy as np
import pandas as pd
import torch


### RNN model

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size,hidden_size,num_layers,nonlinearity='tanh',batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input
        hidden = torch.zeros(self.num_layers, batch_size,self.hidden_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out = self.fc(hidden)
        return out




### LSTM model

class LSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size) #Variable()

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        output, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out) # interesting here is using h_out to feed fc
        return out



### CNN model

class CNN(torch.nn.Module):

    def __init__(self,numFeatures,numLags,numHidden):
        super(CNN, self).__init__()
        self.hidden_size = numHidden
        self.features = numFeatures
        self.num_layers = 'n.a.'
        self.lags = numLags
        self.cov = torch.nn.Conv1d(in_channels=numFeatures,out_channels=numHidden,kernel_size=numLags)
        self.fc = torch.nn.Linear(in_features=numHidden,out_features=1)

    def forward(self,x):
        x = x.view(-1,self.features,self.lags)
        x_ = self.cov(x)
        x_ = x_.view(-1,self.hidden_size)
        x_ = torch.tanh(x_)
        out = self.fc(x_)
        return out




def create_sequences(data,label,lag):
    data_ = []
    label_ = []
    for i in range(len(data)-lag):
        data_seq = data[i:i+lag]
        label_seq = label[i+lag]
        data_.append(data_seq)
        label_.append(label_seq)
    return data_, label_



def create_train_test_datasets(testSize, lag, features=['chg_1d','chg_positive','chg_neutral','chg_negative'], names = ['Tesla', 'Nikola', 'Nio', 'GM', 'Microsoft'], shuffle=False):
    data_train = []
    data_test = []
    label_train = []
    label_test = []

    for name in names:
        file_name = f'output/{name}_processed.csv'
        df = pd.read_csv(file_name)
        data = df[features].to_numpy()
        label = df['chg_1d'].to_numpy().reshape(-1,1)
        data_, label_ = create_sequences(data,label,lag)
        data_train.extend(data_[:-testSize])
        data_test.extend(data_[-testSize:])
        label_train.extend(label_[:-testSize])
        label_test.extend(label_[-testSize:])

    if shuffle:
        indx = np.random.permutation(len(label_train))
        data_train = np.array(data_train)[indx]
        label_train = np.array(label_train)[indx]

    return data_train,label_train,data_test,label_test
