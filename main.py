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
from test import test
#from evaluate import evaluate
from shared_optim import SharedRMSprop, SharedAdam
import time
import copy
import random
from constants import *

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
    default=0.001,
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
    default=2,
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
    help='directory to store trained models')
parser.add_argument(
    '--log-dir',
    type=str,
    default='logs/',
    metavar='LD',
    help='directory to store logs')
parser.add_argument(
    '--epoch',
    type=int,
    default=0,
    metavar='EP',
    help='current epoch, used to pass parameters, do not change')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.96,
    metavar='GM',
    help='to reduce learning rate gradually in simulated annealing')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')
    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    shared_model = SA_NET(Sentence_Max_Length, Embedding_Dim[Tag_Dict[args.tag]])
    if args.model_load:
        try:
            saved_state = torch.load(os.path.join(args.model_dir, 'model_%s.dat' % args.tag))
            shared_model.load_state_dict(saved_state)
        except:
            print('Cannot load existing model from file!')
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()
    
    if args.train:
        while True:
            args.epoch += 1
            print('=====> Train at epoch %d, Learning rate %0.6f <=====' % (args.epoch, args.lr))

            processes = []
            for rank in range(args.workers):
                p = Process(target=train, args=(rank, args, shared_model, optimizer, os.path.join(Dataset_Dir, Tag_Name[Tag_Dict[args.tag]], "%s_train.npz" % Tag_Name[Tag_Dict[args.tag]])))
                p.start()
                processes.append(p)
                time.sleep(0.1)

            for p in processes:
                p.join()

            test(args, shared_model, os.path.join(Dataset_Dir, Tag_Name[Tag_Dict[args.tag]], "%s_test.npz" % Tag_Name[Tag_Dict[args.tag]]))

            args.lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        evaluate()