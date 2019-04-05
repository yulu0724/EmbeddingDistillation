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
# model = inception_v3(dropout=0.5)
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
num_class = len(set(labels))
n_classes = 10
if n_classes > 0:
    features_new = []
    labels_new = []
    classes = [0,6,7,8,9,10,11,12,13,14]
    for feature, label in zip(features, labels):
	if label in classes:
	     features_new.append(feature)
	     labels_new.append(label)
#writer.add_embedding(F.normalize(torch.stack(features_new)), metadata=labels_new)

# export scalar data to JSON for external processing
#writer.export_scalars_to_json("./all_scalars.json")
#writer.close()

# !! --- **** MNI computation is too slow on online-product data set *** --- !! #
# print('compute the NMI index:', NMI(features, labels, n_cluster=num_class))
sim_mat = - pairwise_distance(features)
#np.save('distance_metric',to_numpy(sim_mat))
#np.save('labels',labels)
#pdb.set_trace()
if args.data == 'product':
    print(Recall_at_ks_products(sim_mat, query_ids=labels, gallery_ids=labels))
else:
    print(Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels))




