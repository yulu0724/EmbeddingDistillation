# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance
from evaluations import Recall_at_ks, Recall_at_ks_products
import DataSet
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cub')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument("-gpu",type=str, default='0',help='which gpu to choose')

parser.add_argument('-test', type=int, default=1, help='evaluation on test set or train set')

args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model = torch.load(args.r)
model = model.cuda()

temp = args.r.split('/')
name = temp[-2] + '-' + temp[-1]
if args.test == 1:
    print('test %s***%s' % (args.data, name))
    data = DataSet.create(args.data, train=False)
    data_loader = torch.utils.data.DataLoader(
        data.test, batch_size=8, shuffle=False, drop_last=False)
else:
    print('  train %s***%s' % (args.data, name))
    data = DataSet.create(args.data, test=False)
    data_loader = torch.utils.data.DataLoader(
        data.train, batch_size=8, shuffle=False, drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=32, metric=None)

sim_mat = - pairwise_distance(features)

if args.data == 'product':
    print(Recall_at_ks_products(sim_mat, query_ids=labels, gallery_ids=labels))
else:
    print(Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels))




