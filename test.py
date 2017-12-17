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

def test(args, shared_model, dataset_path):
    start_time = time.time()
    log = setup_logger(0, 'epoch%d_test' % args.epoch, os.path.join(args.log_dir, 'epoch%d_test_log.txt' % args.epoch))
    log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Start testing.')
    local_model = SA_NET(Embedding_Dim[Tag_Dict[args.tag]])
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model = local_model.cuda()

    dataset = np.load(dataset_path)
    targets = dataset["arr_0"]
    correct_cnt = 0

    for idx in range(targets.shape[0]):
        data = dataset["arr_%d" % (idx + 1)]
        if data.shape[0] == 0:
            continue
        data = Variable(torch.from_numpy(data))
        if args.gpu:
            data = data.cuda()
        target = int(targets[idx])
        output = local_model(data)

        '''
        prob = F.softmax(output)
        ans = prob.max(1)[1].data
        if args.gpu:
            ans = ans.cpu()
        ans = ans.numpy()[0]
        if ans == target:
        '''
        if (output.data.cpu().numpy()[0] < 0.5 and target == 0) or (output.data.cpu().numpy()[0] >= 0.5 and target == 1):
            correct_cnt += 1

        if (idx + 1) % 100 == 0:
            log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Accuracy: %d / %d\t%0.4f' % (correct_cnt, idx + 1, correct_cnt / (idx + 1)))
    return correct_cnt / targets.shape[0]
