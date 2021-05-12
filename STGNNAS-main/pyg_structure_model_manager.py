import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

import logging
import os
import argparse
import configparser
import math
import random
import tqdm
import pandas as pd
from sklearn import preprocessing
import utility
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from torchsummary import summary

from structure_model_manager import CitationGNNManager

from utility import data_transform, load_data , load_weighted_adjacency_matrix


def data_preparate(data_path, device, n_his, n_pred, day_slot, batch_size): #数据切片，预处理 √
    data_col = pd.read_csv(data_path, header=None).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = load_data(data_path, len_train, len_val)
    zscore = preprocessing.StandardScaler() #减均值除以标准差
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
    x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
    x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return zscore, train_iter, val_iter, test_iter

def val(model, val_iter, loss): #√
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).reshape(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

class GeoCitationManager(CitationGNNManager):
    def __init__(self, args):
        super(GeoCitationManager, self).__init__(args)

        self.zscore, self.train_iter, self.val_iter, self.test_iter = data_preparate(self.args.data_path, args.device, self.args.n_his, self.args.n_pred, self.args.day_slot, self.args.batch_size)

    def run_model(self, loss, epochs, early_stopping, model, train_iter, val_iter, test_iter, return_best = False): #√
        min_val_loss = np.inf
        best_performance = np.inf
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        for epoch in range(epochs):
            l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
            model.train()
            # for x, y in tqdm.tqdm(train_iter):
            for x, y in train_iter:
                y_pred = model(x).reshape(len(x), -1)
                l = loss(y_pred, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step() 
                scheduler.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            val_loss = val(model, val_iter, loss)
            test_loss = val(model, test_iter, loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), self.args.model_save_path)
                if test_loss < best_performance:
                    best_performance = test_loss

            early_stopping(val_loss, model)
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | Best performance: {:.6f} | GPU occupy: {:.6f} MiB'.\
                format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, best_performance, gpu_mem_alloc))
            if early_stopping.early_stop:
                print("Early stopping.")
                break
        print('\nTraining finished.\n')

        if return_best:
            return model, val_loss, best_performance
        else:
            return model, val_loss

