from __future__ import print_function, division
import os
import argparse
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Value
from model import SA_NET
from train import train
#from test import test
#from evaluate import evaluate
from shared_optim import SharedRMSprop, SharedAdam
import time
import copy
import random

parser = argparse.ArgumentParser(description='Sentiment-Analysis')
parser.add_argument(
    '--train',
    default = True,
    metavar = 'T',
    help = 'train model (set False to evaluate)')
parser.add_argument(
    '--gpu',
    default=True,
    metavar='G',
    help='using GPU')
parser.add_argument(
    '--model-load',
    default=False,
    metavar='L',
    help='load trained model')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate')
parser.add_argument(
    '--seed',
    type=int,
    default=233,
    metavar='S',
    help='random seed')
parser.add_argument(
    '--workers',
    type=int,
    default=12,
    metavar='W',
    help='how many training processes to use')
parser.add_argument(
    '--tag',
    type=str,
    default='EN',
    metavar='TG',
    help='language of corpus')
parser.add_argument(
    '--model-dir',
    type=str,
    default='trained_models/',
    metavar='MD',
    help='folder to store trained models')
parser.add_argument(
    '--epoch',
    type=int,
    default=0,
    metavar='EP',
    help='current epoch, used to pass parameters, do not change')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')
    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    shared_model = SA_NET(256, 256)
    if args.model_load:
        try:
            saved_state = torch.load(os.path.join(args.model_dir, 'model_%s.dat' % args.tag))
            shared_model.load_state_dict(saved_state)
        except:
            print('Cannot load existing model from file!')
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()
    if not os.path.exists('dataset/cn/cn_negative.npy'):
        initialize
    
    if args.train:
        while True:
            args.epoch += 1
            print('=====> Train at epoch %d <=====' % args.epoch)

            processes = []
            for rank in range(args.workers):
                p = Process(target=train, args=(rank, args, shared_model, optimizer, np.zeros((100, 100, 256, 256))))
                p.start()
                processes.append(p)
                time.sleep(0.1)

            for p in processes:
                p.join()