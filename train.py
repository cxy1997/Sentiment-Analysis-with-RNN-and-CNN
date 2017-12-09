from __future__ import print_function, division
import os
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
from constants import *

def train(rank, args, shared_model, optimizer, dataset_path):
    start_time = time.time()
    log = setup_logger(rank, 'epoch%d_worker%d' % (args.epoch, rank), os.path.join(args.log_dir, 'epoch%d_worker%d_log.txt' % (args.epoch, rank)))
    log.info('Train time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Training started.')
    local_model = SA_NET(Sentence_Max_Length, Embedding_Dim[Tag_Dict[args.tag]])
    local_model.train()
    criterion = nn.CrossEntropyLoss()

    dataset = np.load(dataset_path)
    targets = dataset["arr_0"]
    order = list(range(targets.shape[0]))
    random.shuffle(order)
    losses = 0

    for i in range(targets.shape[0]):
        idx = order[i]
        local_model.load_state_dict(shared_model.state_dict())
        if args.gpu:
            local_model = local_model.cuda()
            cx = Variable(torch.zeros(1, 512).cuda())
            hx = Variable(torch.zeros(1, 512).cuda())
        else:
            cx = Variable(torch.zeros(1, 512))
            hx = Variable(torch.zeros(1, 512))

        data = dataset["arr_%d" % (idx + 1)]
        target = Variable(torch.LongTensor([int(targets[idx])]), requires_grad = False)
        #print(idx, data.shape, target.data.numpy()[0])
        if args.gpu:
            target = target.cuda()

        for j in range(data.shape[0]):
            feed_data = torch.from_numpy(data[j]).unsqueeze(0).unsqueeze(0).float()
            if args.gpu:
                feed_data = feed_data.cuda()
            output, (hx, cx) = local_model((Variable(feed_data),(hx, cx)))

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        if args.gpu:
            loss = loss.cpu()

        if args.gpu:
            local_model = local_model.cpu()
        ensure_shared_grads(local_model, shared_model)
        optimizer.step()
        losses += loss
        if (i + 1) % 10 == 0:
            log.info('Train time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Mean loss: %0.4f' % (loss.data.numpy()[0] % 100))