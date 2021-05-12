import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


from utility import TopSmallAverage 


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):

    def __init__(self, args):
        self.args = args
        self.reward_manager = TopSmallAverage(10)

        self.args = args
        self.drop_out = args.drop_rate
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay_rate
        
        self.retrain_epochs = args.retrain_epochs
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = nn.MSELoss()


    # train from scratch
    def evaluate(self, actions=None, format="two"):
        print("All parameters in model for evaluating is here:")
        print(actions)

        # create model
        model = self.build_gnn(actions) #来自micro_model_manager！
        if self.args.cuda:
            model.cuda()
        
        try:
            model, val_loss, test_loss = self.run_model(self.loss_fn, self.epochs, self.args.early_stopping, model, self.train_iter, self.val_iter, self.test_iter, return_best=True)

        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_loss = 0
                test_loss = 0
            else:
                raise e
        return val_loss, test_loss

    # train from scratch
    def train(self, actions=None, format="two"):
        print("All parameters in model for training is here:")
        print(actions)
        origin_action = actions
        print("train action:", actions) 

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            
            model, val_loss = self.run_model(self.loss_fn, self.epochs, self.args.early_stopping, model, self.train_iter, self.val_iter, self.test_iter)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_loss = 0
            else:
                raise e
                
        if len(self.args.all_loss_list) > 0:
            avg = np.mean(self.args.all_loss_list)
        else:
            avg = 0
            
        self.args.all_loss_list.append(val_loss)
        self.args.all_loss_list.sort(reverse=False)
        self.args.all_loss_list = self.args.all_loss_list[:10] #self.topsmall_k = 10
        reward = (avg - val_loss)/10
        
        print(f"The current reward is: {reward}")
        self.record_all_action_info(origin_action, reward, val_loss) 

        return reward, val_loss

    def record_all_action_info(self, origin_action, reward, val_loss): 
        
        with open(self.args.dataset + "_" +  self.args.submanager_log_file, "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_loss))
            file.write("\n")

    def retrain(self, actions, format="two"):
        return self.train(actions, format)

    def test_with_param(self, actions=None, format="two", with_retrain=False):#计算reward用
        return self.train(actions, format)

