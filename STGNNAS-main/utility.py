from __future__ import print_function

import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.autograd import Variable

import pandas as pd

from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigsh


def calculate_laplacian_matrix(adj_mat, mat_type): #输入 矩阵(np.array) 和 矩阵类型(如'id_mat') 输出相应矩阵

    n_vertex = adj_mat.shape[0]
    id_mat = np.asmatrix(np.identity(n_vertex))

    # D_row
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # D_com
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))

    # D = D_row as default
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)
    
    deg_mat_inv = np.linalg.inv(deg_mat)
    deg_mat_inv[np.isinf(deg_mat_inv)] = 0.

    deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
    deg_mat_inv_sqrt[np.isinf(deg_mat_inv_sqrt)] = 0.

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    wid_deg_mat_inv = np.linalg.inv(wid_deg_mat)
    wid_deg_mat_inv[np.isinf(wid_deg_mat_inv)] = 0.

    wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)
    wid_deg_mat_inv_sqrt[np.isinf(wid_deg_mat_inv_sqrt)] = 0.

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    # Symmetric normalized Laplacian
    # For SpectraConv
    # To [0, 1]
    # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
    sym_normd_lap_mat = id_mat - np.matmul(np.matmul(deg_mat_inv_sqrt, adj_mat), deg_mat_inv_sqrt)

    # For ChebConv
    # From [0, 1] to [-1, 1]
    # wid_L_sym = 2 * L_sym / lambda_max_sym - I
    sym_max_lambda = max(np.linalg.eigh(sym_normd_lap_mat)[0])
    #sym_max_lambda = eigsh(sym_normd_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
    wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / sym_max_lambda - id_mat

    # For GCNConv
    # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
    hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

    # Random Walk normalized Laplacian
    # For SpectraConv
    # To [0, 1]
    # L_rw = D^{-1} * L_com = I - D^{-1} * A
    rw_normd_lap_mat = id_mat - np.matmul(deg_mat_inv, adj_mat)

    # For ChebConv
    # From [0, 1] to [-1, 1]
    # wid_L_rw = 2 * L_rw / lambda_max_rw - I
    rw_max_lambda = max(np.linalg.eigh(rw_normd_lap_mat)[0])
    #rw_max_lambda = eigsh(rw_normd_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
    wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / rw_max_lambda - id_mat

    # For GCNConv
    # hat_L_rw = wid_D^{-1} * wid_A
    hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat
    elif mat_type == 'sym_normd_lap_mat':
        return sym_normd_lap_mat
    elif mat_type == 'wid_sym_normd_lap_mat':
        return wid_sym_normd_lap_mat
    elif mat_type == 'hat_sym_normd_lap_mat':
        return hat_sym_normd_lap_mat
    elif mat_type == 'rw_normd_lap_mat':
        return rw_normd_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

def evaluate_model(model, loss, data_iter): #输入 model 和 loss_functuon 和 data_iterator(里面包含了所有信息)，输出 均方误差
    model.eval() #model.eval()在测试模型时在前面使用，有固定的使用场景
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler): #输入 model 和 data_iterator(里面包含了所有信息) 和 scaler(后面为zscore：减均值除以标准差), 输出 MAE, RMSE, WMAPE
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, RMSE, WMAPE

def data_transform(data, n_his, n_pred, day_slot, device):#√
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def load_data(file_path, len_train, len_val):#√
    df = pd.read_csv(file_path, header=None)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test

def load_weighted_adjacency_matrix(file_path):#√
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


class TopSmallAverage(object): 
    def __init__(self, topsmall_k=10):
        self.scores = []
        self.topsmall_k = topsmall_k

    def get_topsmall_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[self.topsmall_k:]
        return avg

    def get_reward(self, score):

        print('Score and self.get_average(score) is:: {:.10f} , {:.20f}'.\
            format(score, self.get_average(score)))
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)


def process_action(actions, type, args):

    if type == 'two':
        actual_action = actions
        actual_action[-1] = args.num_class

        return actual_action

    elif type == "simple":
        actual_action = actions
        index = len(actual_action) - 1
        actual_action[index]["out_dim"] = args.num_class

        return actual_action

    elif type == "dict":
        return actions

    elif type == "micro":
        return actions





class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py 
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def prepare_dirs(args):
    """Sets the directories for the model, and creates those directories.

    Args:
        args: Parsed from `argparse` in the `config` module.
    """
    if args.load_path:
        if args.load_path.startswith(args.log_dir):
            args.model_dir = args.load_path
        else:
            if args.load_path.startswith(args.dataset):
                args.model_name = args.load_path
            else:
                args.model_name = "{}_{}".format(args.dataset, args.load_path)
    else:
        args.model_name = "{}_{}".format(args.dataset, get_time())

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)
    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def save_dag(args, dag, name):
    save_path = os.path.join(args.model_dir, name)
    logger.info("[*] Save dag : {}".format(save_path))
    json.dump(dag, open(save_path, 'w'))


def load_dag(args):
    load_path = os.path.join(args.dag_path)
    logger.info("[*] Load dag : {}".format(load_path))
    with open(load_path) as f:
        dag = json.load(f)
    return dag


def makedirs(path):#在path下建立一个文件夹
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)


def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)


def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()

