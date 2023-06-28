#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from utils.quantization import sr_nn,randomk_nn #引入量化稀疏化
from utils.quantization import topk_nn
##from sklearn import metrics##作用不明


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.q_update = None  # 模型更新参数

    def train(self, net):
        net.train()  # nn.Module.train：把本层和子层的training设置为true--使用BatchNormalizetion()和Dropout()
        # train and update
        server_net = copy.deepcopy(net)  # 保存初始网络
        self.q_update = copy.deepcopy(net)  # 初始化q_update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # print(net.parameters())##打印网络参数

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # print("log_probs")打印属性
                # print(log_probs)
                # print("labels")##打印标签
                # print(labels)
                loss = self.loss_func(log_probs, labels)
                # print("loss function")
                # print(loss)##打印loss
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.compute_update(initial_model=server_net, updated_model=net)  # 计算模型差值
        if self.args.comm_scheme == "spar":
            self.q_update = randomk_nn(self.q_update, self.args.q_spar)  # 稀疏化
        elif self.args.comm_scheme == "topk":
            self.q_update = topk_nn(self.q_update,self.args.q_spar)
        self.q_update = sr_nn(self.q_update, self.args.q_quan)  # 量化
        return self.q_update.state_dict(), sum(epoch_loss) / len(epoch_loss)  # 返回client更新后模型

    def compute_update(self, initial_model, updated_model):
        initial_param = list(initial_model.parameters())
        updated_param = list(updated_model.parameters())
        gradient_param = list(self.q_update.parameters())
        nc = len(initial_param)
        for i in range(nc):
            gradient_param[i].data[:] = updated_param[i].data[:] - initial_param[i].data[:]
        return None

