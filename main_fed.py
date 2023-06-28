#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
from torchvision import datasets, transforms
import torch
import datetime
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Nets import resnet18
from models.Fed import FedAvg
from models.test import test_img
from models.ini_loss import compute_loss
from models.param_length import param_length
import numpy as np


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        test_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar = transforms.Compose([transforms.Resize(40), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32),
                                          transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=test_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'cifar-100':
        test_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar = transforms.Compose([transforms.Resize(40), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32),
                                          transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar-100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar-100', train=False, download=True, transform=test_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet18':
        net_glob = resnet18().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    print(datetime.datetime.now())  # 打印开始时间
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    # initial loss
    loss_0 = compute_loss(net_glob, dataset_train, args)

    # dimension of parameter
    length = param_length(model=copy.deepcopy(net_glob).to(args.device))

    # training
    acc_acuracy = []  # 画准确率图
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    total_train = 0  # 总迭代轮次
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if args.comm_scheme == 'local-SGD':  # 本地更新为8个epoch 每个epoch 16次本地更新
        args.epochs = args.epochs//5
        print(args.epochs)
    elif args.comm_scheme == 'adaptiveQSGD':
        args.epochs = 11
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))  # w：client更新后模型
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)
        # 更新server参数
        sd = net_glob.state_dict()
        for key in sd.keys():
            sd[key] = torch.add(net_glob.state_dict()[key], w_glob[key])
        # copy weight to net_glob
        net_glob.load_state_dict(sd)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # 画acc_accuracy
        if args.comm_scheme != 'adaptiveQSGD' and (iter+1)%10 == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            acc_acuracy.append(acc_test)
            loss_train.append(loss_test)
            print('Round {:3d}, test accuracy {:.3f}'.format(iter, acc_test))
        elif args.comm_scheme != 'adaptiveQSGD' and iter == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            loss_train.append(loss_test)
            acc_acuracy.append(acc_test)
            print('Round {:3d}, test accuracy {:.3f}'.format(iter, acc_test))
        elif args.comm_scheme == 'adaptiveQSGD':
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            acc_acuracy.append(acc_test)
            loss_train.append(loss_test)
            print('Round {:3d}, test accuracy {:.3f}'.format(iter, acc_test))
        if args.comm_scheme == 'adaptiveQSGD':
            # Adaptive sgd
            print(iter)
            alpha = 0.95
            # alpha = (loss_avg/loss_0)**(1/(iter+1))  # 计算alpha,iter从0开始取值
            # print("alpha:")
            # print(alpha)
            # 计算本地更新轮次
            total_iter = 300
            comm_round = 10
            total_train += args.local_ep
            args.local_ep = math.sqrt(((1-alpha**(1/2))**2)*(total_iter**2)/((alpha**(comm_round-iter))*(alpha**(-comm_round/2)-1)**2))
            args.local_ep=round(args.local_ep)
            print(args.local_ep)
            if args.local_ep<1:
                args.local_ep=1
            args.q_quan = math.log2(2*math.log(2)*(args.local_ep*length*2-32))/2  # quantization bit
            k = (args.local_ep * length*2 - 32) / args.q_quan   # sparcification level
            args.q_quan = math.floor(args.q_quan)
            args.q_spar = k/length
            # if iter>=20:
               #  args.lr=0.001
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        elif args.comm_scheme == 'quan':
            args.q_quan = 2
            total_train += args.local_ep
        elif args.comm_scheme == 'spar':
            args.q_spar = 0.0625
            total_train += args.local_ep
        elif args.comm_scheme == 'topk':
            args.q_spar = 0.0625
            total_train += args.local_ep
        elif args.comm_scheme == 'AC-SGD':
            args.q_quan = math.log2(2*math.log(2)*(length*2-32))/2
            args.q_spar = (2*length-32)/args.q_quan
            total_train += args.local_ep
        elif args.comm_scheme == 'local-SGD':
            args.local_ep = 5
            args.q_quan = 2*args.local_ep
            total_train += args.local_ep
        elif args.comm_scheme == 'SGD':
            total_train += args.local_ep
        else :
            print("algorithm error")

    print("总训练轮次")
    print(total_train)


    # plot acc_test curve

    plt.figure()
    plt.plot(range(len(acc_acuracy)), acc_acuracy)
    plt.ylabel('acc_test')
    plt.savefig('./save/fed_{}_{}_ep{}_localep{}_C{}_iid{}_{}_total_train{}_momentum{}_data-aug.png'.format(args.dataset, args.model, args.epochs, args.local_ep, args.frac, args.iid, args.comm_scheme, total_train, args.momentum))
    accuracy = torch.tensor(acc_acuracy)
    loss_save = torch.tensor(loss_train)
    torch.save(accuracy, './save/train_data/algorithm_{}_dataset_{}_model_{}.txt'.format(args.comm_scheme, args.dataset, args.model))
    torch.save(loss_save, './save/train_data/algorithm_{}_dataset_{}_model_{}_loss.txt'.format(args.comm_scheme, args.dataset, args.model))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print(datetime.datetime.now())  # 打印结束时间

