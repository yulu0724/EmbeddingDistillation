# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display
import DataSet
import pdb
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cudnn.benchmark = True


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dir = '%s_%s_dis_%s_%s_%s_%0.2f_%s' % (args.data, args.loss, args.net,args.TNet, args.Ttype, args.lamda,args.lr)
    log_dir = os.path.join('checkpoints', dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))
    display(args)
    # Teacher Netowrk
    if args.r is None:
        Network_T = args.TNet
        model_T = models.create(Network_T, Embed_dim=args.dim)
        model_dict_T = model_T.state_dict()
        
        if args.data == 'cub':
            model_T = torch.load('checkpoints/cub_Tmodel.pkl') 
        elif args.data == 'car':
            model_T = torch.load('checkpoints/car_Tmodel.pkl')
        elif args.data == 'product':
            model_T = torch.load('checkpoints/product_Tmodel.pkl')

    else:
        model_T = torch.load(args.r)
    
    model_T = model_T.cuda()
    model_T.eval()

    # Student network
    if args.r is None:
        model = models.create(args.net, Embed_dim=args.dim)
        model_dict = model.state_dict()
        if args.net == 'bn':
            pretrained_dict = torch.load('pretrained_models/bn_inception-239d2248.pth')
        elif args.net == 'resnet101':
            pretrained_dict = torch.load('pretrained_models/resnet101-5d3b4d8f.pth')
        elif args.net == 'resnet50':
            pretrained_dict = torch.load('pretrained_models/resnet50-19c8e357.pth')
        elif args.net == 'resnet34':
            pretrained_dict = torch.load('pretrained_models/resnet34-333f7ec4.pth')
        elif args.net == 'resnet18':
            pretrained_dict = torch.load('pretrained_models/resnet18-5c106cde.pth')
        elif args.net == 'inception':
            pretrained_dict = torch.load('pretrained_models/inception_v3_google-1a9a5a14.pth')
        else:
            print (' Oops!  That was no valid models. ')


        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    else:
        model = torch.load(args.r)
    
    if args.continue_train:
        model=torch.load(log_dir+'/%d_model.pkl' % (args.start))
    
    model = model.cuda()
     
    torch.save(model, os.path.join(log_dir, 'model.pkl'))
    print('initial model is save at %s' % log_dir)

    new_param_ids = set(map(id, model.Embed.parameters()))

    new_params = [p for p in model.parameters() if
                   id(p) in new_param_ids]

    base_params = [p for p in model.parameters() if
                    id(p) not in new_param_ids]
    param_groups = [
                 {'params': base_params, 'lr_mult': 0.1},
                 {'params': new_params, 'lr_mult': 1.0}]


    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.loss == 'knnsoftmax':
        criterion = losses.create(args.loss, alpha=args.alpha, k=args.k).cuda()
    else:
        criterion = losses.create(args.loss).cuda()

    data = DataSet.create(args.data, root=None, test=False)
    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.BatchSize,
        sampler=RandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, num_workers=args.nThreads)

    loss_log=[]
    for i in range(3):
        loss_log.append([]) 
    loss_dis = []
    for i in range(3):
        loss_dis.append([])
    for epoch in range(args.start, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
        
            embed_feat = model(inputs)
            embed_feat_T = model_T(inputs)

            loss_net, inter_, dist_ap, dist_an, dis_pos, dis_neg,dis = criterion(embed_feat, labels)
            loss_net_T, inter_T, dist_ap_T, dist_an_T,dis_pos_T, dis_neg_T,dis_T = criterion(embed_feat_T, labels)

            lamda=args.lamda
            
            if args.Ttype == 'relative':
                loss_dis[0].append(torch.mean(torch.norm(dis-dis_T,p=2)).data[0])
                loss_dis[1].append(0.0)
                loss_dis[2].append(0.0)
                
                loss_distillation = 0.0*torch.mean(F.pairwise_distance(embed_feat,embed_feat_T))
                loss_distillation += torch.mean(torch.norm(dis-dis_T,p=2))
                loss = loss_net + lamda * loss_distillation
            
            elif args.Ttype == 'absolute':
                loss_dis[0].append(0.0)
                loss_dis[1].append(0.0)
                loss_dis[2].append(torch.mean(F.pairwise_distance(embed_feat,embed_feat_T)).data[0])
                loss_distillation = torch.mean(F.pairwise_distance(embed_feat,embed_feat_T))
                loss = loss_net + lamda * loss_distillation

            else:
                print('This type does not exist')
    
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]        
            loss_log[0].append(loss.data[0])
            loss_log[1].append(loss_net.data[0])
            loss_log[2].append(lamda*loss_distillation.data[0])

            if epoch == 0 and i == 0:
                print(50*'#')
                print('Train Begin -- HA-HA-HA')
        
        print('[Epoch %05d]\t  Loss_net: %.3f \t Loss_distillation: %.3f \t Accuracy: %.3f \t Pos-Dist: %.3f \t Neg-Dist: %.3f' % (epoch + 1,  loss_net, lamda*loss_distillation, inter_, dist_ap, dist_an))

        


        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(log_dir, '%d_model.pkl' % epoch))

    #plot loss 
    line1,=plt.plot(loss_log[0],'r-',label="Total loss",)
    line2,=plt.plot(loss_log[1],'b-',label = "KNNsoftmax loss")
    line3,=plt.plot(loss_log[2],'g--',label ="Distillation loss")
    plt.title('%s_%s_dis_%s_%s_%s_%0.2f' % (args.data, args.loss, args.net,args.TNet, args.Ttype, args.lamda))
    plt.legend([line1,line2,line3],['Total loss','Contrastive loss', 'Distance loss'])
    plt.savefig('./fig/%s_%s_dis_%s_%s_%s_%0.2f.jpg' % (args.data, args.loss, args.net,args.TNet, args.Ttype, args.lamda))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN-Softmax Training')

    # hype-parameters
    parser.add_argument('-lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('-num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('-dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('-alpha', default=40, type=int, metavar='n',
                        help='hyper parameter in KNN Softmax')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')

    # network
    parser.add_argument('-data', default='cub', required=True,
                        help='path to Data Set')
    parser.add_argument('-net', default='bn')
    parser.add_argument('-loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('-epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('-save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    parser.add_argument('-r', default=None,
                        help='the path of the pre-trained model')
    parser.add_argument('-start', default=0, type=int,
                        help='resume epoch')

    # basic parameter
    parser.add_argument('-log_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)

    # distillation
    parser.add_argument("-lamdaA",type=float, default=0.0, help="The trade-off between contrastive and other losses") 
    parser.add_argument("-lamdaB",type=float, default=0.0, help="The trade-off between contrastive and other losses") 
    parser.add_argument("-lamda",type=float, default=0.0, help="The trade-off between contrastive and other losses") 

    parser.add_argument("-Ttype",type=str, default='D', help='relative, absolute')
    parser.add_argument('-which_epoch', type=int, default=0, help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('-continue_train', action='store_true', help='continue training: load the certain model')
    parser.add_argument("-gpu",type=str, default='0',help='which gpu to choose')
    parser.add_argument("-TNet",type=str, default='resnet101',help='which teacher model to choose')

    main(parser.parse_args())




