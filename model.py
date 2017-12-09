from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init

class SA_NET(torch.nn.Module):
    def __init__(self, sentence_length, embedding_length, classes = 2):
        super(SA_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 1024, (7, embedding_length), stride = 1, padding = (3, 0))
        self.conv2 = nn.Conv1d(1024, 64, 9, stride = 1, padding = 4)
        self.conv3 = nn.Conv1d(64, 128, 5, stride = 1, padding = 2)
        self.conv4 = nn.Conv1d(128, 256, 3, stride = 1, padding = 1)

        self.lstm = nn.LSTMCell(sentence_length * 16, 512)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, classes)

        self.apply(weights_init)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(F.max_pool1d(self.conv1(inputs).squeeze(3), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv4(x), kernel_size=2, stride=2))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = F.relu(self.fc1(hx))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x, (hx, cx)

if __name__ == '__main__':
	c = SA_NET(128, 100)
	print(c.forward((Variable(torch.ones(1, 1, 128, 100)), (Variable(torch.zeros(1, 512)), Variable(torch.zeros(1, 512))))))