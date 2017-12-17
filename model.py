from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init
from constants import *

class SA_NET(torch.nn.Module):
    def __init__(self, embedding_length, classes = 2):
        super(SA_NET, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, (7, embedding_length), stride = 1, padding = (3, 0))
        self.conv2 = nn.Conv1d(256, 64, 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv1d(64, 256, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv1d(256, 16, 1, stride = 1, padding = 0)

        self.lstm = nn.LSTMCell(embedding_length, LSTM_Hidden_Size)

        self.fc1 = nn.Linear(LSTM_Hidden_Size + CNN_Feature_Size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.apply(weights_init)

        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        if next(self.parameters()).is_cuda:
            if not inputs.is_cuda:
                inputs = inputs.cuda()
            cx = Variable(torch.zeros(1, LSTM_Hidden_Size).cuda())
            hx = Variable(torch.zeros(1, LSTM_Hidden_Size).cuda())
            hxs = Variable(torch.zeros((inputs.size()[0], LSTM_Hidden_Size)).cuda())
            x0 = Variable(torch.zeros(1, 1, Sentence_Max_Length, inputs.size(1)).cuda())
        else:
            if inputs.is_cuda:
                inputs = inputs.cpu()
            cx = Variable(torch.zeros(1, LSTM_Hidden_Size))
            hx = Variable(torch.zeros(1, LSTM_Hidden_Size))
            hxs = Variable(torch.zeros((inputs.size()[0], LSTM_Hidden_Size)))
            x0 = Variable(torch.zeros(1, 1, Sentence_Max_Length, inputs.size(1)))

        for i in range(inputs.size()[0]):
            hx, cx = self.lstm(inputs[i], (hx, cx))
            hxs[i] = hx
        
        hx_mean = torch.mean(hxs, 0, True)

        inputs = inputs.unsqueeze(0).unsqueeze(0)
        x0[:, :, :min(Sentence_Max_Length, inputs.size(2)),:] = inputs[:, :, :min(Sentence_Max_Length, inputs.size(2)),:]
        x0 = F.sigmoid(F.max_pool1d(self.conv1(x0).squeeze(3), kernel_size=2, stride=2))
        x0 = F.sigmoid(F.max_pool1d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = F.sigmoid(F.max_pool1d(self.conv3(x0), kernel_size=2, stride=2))
        x0 = F.sigmoid(self.conv4(x0))
        
        x0 = x0.view(x0.size(0), -1)
        
        x = torch.cat((hx_mean, x0), 1)
        #print(x)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x#F.log_softmax(x)

if __name__ == '__main__':
    c = SA_NET(256).cuda()
    print(c.forward(Variable(torch.ones(5, 256)).cuda())[0].size())
