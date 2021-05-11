import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn.conv import *

class IncrementSearchSpace(object):
    def __init__(self, search_space=None, max_cell=2):
        self.search_space = {}
        self.search_space["act"] = [ "sigmoid", "tanh", "softsign", "relu", "softplus", "leakyrelu", "prelu", "elu" ] #8
        self.search_space['learning_rate'] = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4 ] #5
        self.search_space['dropout'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] #10
        self.search_space['weight_decay'] = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0 ] #6
        self.search_space['mat_type'] = [ "rw_rw", "rw_sym", "sym_rw", "sym_sym" ] #4
        self.search_space['gated_act_func'] = [ "glu", "gtu" ] #2
        self.search_space['ratio'] = [ "three_one", "two_one", "three_two", "one_one", "two_three", "one_two", "one_three" ] #7
        self.search_space['graph_conv_type'] = [ "chebconv", "gcnconv" ] #2
        for i in range(1):
            self.search_space[f"self_index_{i}"] = list(range(2+i))  
        pass

    def get_search_space(self):
        return self.search_space
    
    @staticmethod
    def generate_action_list(): 
        action_list = ['self_index_0', 'graph_conv_type', 'gated_act_func', 'ratio', 'act', 'dropout', 'mat_type', 'learning_rate', 'weight_decay']
        return action_list


def act_map(act):
    if act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "relu":
        return nn.ReLU()
    elif act == "softplus":
        return nn.Softplus()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    elif act == "prelu":
        return nn.PReLU()
    elif act == "elu":
        return nn.ELU()
    else:
        raise Exception("wrong activate function")
