from __future__ import print_function, division
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import setup_logger, ensure_shared_grads
from model import SA_NET
from torch.autograd import Variable
from shared_optim import SharedRMSprop, SharedAdam
import torch.nn as nn
import time
import random
import numpy as np
import logging
import copy

def train(rank, args, shared_model, optimizer, dataset):
    local_model = SA_NET(256, 256)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    local_dataset = copy.deepcopy(dataset)
    np.random.shuffle(local_dataset)