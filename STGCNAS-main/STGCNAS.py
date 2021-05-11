#encoding="utf-8"
from __future__ import print_function
import logging
import os
import argparse
import configparser
import math
import random
import tqdm
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigsh
import pandas as pd
from sklearn import preprocessing
import time
import json
from collections import defaultdict
from datetime import datetime

import torch

import torch.nn.functional as F
import torch.optim as optim
import tensor_utils as utils
import torch.nn.init as init
from torch.autograd import Variable
from torchsummary import summary

import trainer




def build_args(): #传参用
    parser = argparse.ArgumentParser(description='STGCNAS for road traffic prediction')
    register_default_args(parser)
    args = parser.parse_args()
    return args

def register_default_args(parser):#定义默认值

    parser.add_argument('--cuda', type=bool, default='True', help='要不要CUDA，默认为True') 

    parser.add_argument('--random_seed', type=int, default=1608825600)

    parser.add_argument('--save_epoch', type=int, default=2)

    parser.add_argument('--max_save_num', type=int, default=5)

    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])

    parser.add_argument('--derive_from_history', type=bool, default=True)

    parser.add_argument('--opt', type=str, default='AdamW', help='使用的optimizer, 默认为AdamW')

    parser.add_argument('--controller_grad_clip', type=float, default=0)

    parser.add_argument('--n_his', type=int, default='12', help='输入时间，默认为12 (60min)')

    parser.add_argument("--time_intvl", type=int, default="5")

    parser.add_argument("--Kt", type=int, default="3")

    parser.add_argument("--stblock_num", type=int, default="2", help="要求((Kt - 1) * 2 * stblock_num <= n_his) and ((Kt - 1) * 2 * stblock_num > 0)")

    parser.add_argument("--drop_rate", type=float, default="0.6", help="搜索的一部分，等同于IncrementSearchSpace中的dropout")

    parser.add_argument("--multi_label", type=bool, default=False, help="multi_label or single_label task")

    parser.add_argument('--batch_size', type=int, default=32, help="每部分大小，STGCN默认为32")

    parser.add_argument("--lr", type=float, default=0.005, help="搜索的一部分，等同于IncrementSearchSpace中的learning_rate")

    parser.add_argument('--controller_lr', type=float, default=3.5e-4, help="will be ignored if --controller_lr_cosine=True")

    parser.add_argument('--tanh_c', type=float, default=2.5)

    parser.add_argument('--softmax_temperature', type=float, default=5.0)

    parser.add_argument('--weight_decay_rate', type=float, default=5e-4, help="搜索的一部分，等同于IncrementSearchSpace中的weight_decay")

    parser.add_argument('--step_size', type=int, default=10, help="scheduler里需要")

    parser.add_argument('--gamma', type=float, default=0.999, help="scheduler里需要")

    parser.add_argument('--entropy_coeff', type=float, default=1e-4)

    parser.add_argument("--param_file", type=str, default="pemsd7-m_test.pkl",
                        help="learning rate")
                        
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")

    parser.add_argument("--optim_file", type=str, default="opt_pemsd7-m_test.pkl",
                        help="optimizer save path")

    parser.add_argument('--derive_finally', type=bool, default=True)

    #dataset部分
    parser.add_argument("--dataset", type=str, default="pemsd7-m", required=False, help="输入数据集，可选择三种不同数据集，默认为pemsd7-m")

    parser.add_argument('--data_path', type=str, default="./data/train/road_traffic/pemsd7-m/vel.csv", help="dataset路径，和上面一起改")

    parser.add_argument('--wam_path', type=str, default="./data/train/road_traffic/pemsd7-m/adj_mat.csv", help="wam路径，和上面一起改")

    parser.add_argument('--model_save_path', type=str, default="./model/save/road_traffic/pemsd7-m/", help="保存model的路径，和上面一起改")


    #影响运行时间部分+重要部分⭐
    #sub_manager_logger_file打印条数：max_epoch * ( 1 + retrain_epochs )

    parser.add_argument('--n_pred', type=int, default=12, help='预测时间长度，6的话是30min！')

    parser.add_argument('--max_epoch', type=int, default=10) 

    parser.add_argument('--epochs', type=int, default=5, help='训练次数，STGCN默认为50，（这里先2）') #

    parser.add_argument("--retrain_epochs", type=int, default=5,
                        help="number of training epochs")

    parser.add_argument('--derive_num_sample', type=int, default=2) #⭐️

    parser.add_argument('--controller_max_step', type=int, default=1, help='step for controller parameters') #controller_max_step ⭐️


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False #有没有cuda
        
    torch.manual_seed(args.random_seed)#为CPU设置种子用于生成随机数，以使得结果是确定的 
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed) #torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子

    utils.makedirs(args.dataset) #在当前目录下建立一个以dataset为名的文件夹
    
    trnr = trainer.Trainer(args)#按照args的参数开始

    trnr.train()


if __name__ == "__main__":
    args = build_args() #args按照上面的parser.add_argument定义内容
    main(args) #开始按照args内容跑



