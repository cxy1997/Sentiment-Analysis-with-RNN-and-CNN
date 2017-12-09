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
    local_model = SA_NET(Sentence_Max_Length, Embedding_Dim[Tag_Dict[args.tag]])
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model = local_model.cuda()

    dataset = np.load(dataset_path)
    targets = dataset["arr_0"]
    correct_cnt = 0

    for idx in range(targets.shape[0]):
        if args.gpu:
            cx = Variable(torch.zeros(1, 512).cuda())
            hx = Variable(torch.zeros(1, 512).cuda())
        else:
            cx = Variable(torch.zeros(1, 512))
            hx = Variable(torch.zeros(1, 512))

        data = dataset["arr_%d" % (idx + 1)]
        target = int(targets[idx])

        for j in range(data.shape[0]):
            feed_data = torch.from_numpy(data[j]).unsqueeze(0).unsqueeze(0).float()
            if args.gpu:
                feed_data = feed_data.cuda()
            output, (hx, cx) = local_model((Variable(feed_data),(hx, cx)))

        prob = F.softmax(output)
        ans = prob.max(1)[1].data
        if args.gpu:
            ans = ans.cpu()
        ans = ans.numpy()[0]
        print(ans, target)
        if ans == target:
            print("yeah")
            correct_cnt += 1

        if (idx + 1) % 10 == 0:
            log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Accuracy: %d / %d\t%0.4f' % (correct_cnt, idx + 1, correct_cnt / (idx + 1)))