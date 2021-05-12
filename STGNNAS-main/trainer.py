import glob
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import tensor_utils
import earlystopping
from earlystopping import EarlyStopping

logger = tensor_utils.get_logger()

from sklearn import preprocessing
from structure_model_manager import CitationGNNManager
import logging
import os
import argparse
import configparser
import math
import random
import tqdm
import pandas as pd
from micro_model_manager import *
from utility import *
from micro_search_space import *


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value

def _get_optimizer(name): #使用不同的optimizer
    if name.lower() == 'sgd':
        optim = torch.optim.SGD #根据的是一整个数据集的随机一部分计算它们所有的梯度，然后执行决策
    elif name.lower() == 'adam':
        optim = torch.optim.Adam
    elif name.lower() == 'adamw':
        optim = torch.optim.AdamW
    elif name.lower() == 'rmsprop':
        optim = torch.optim.RMSProp
    else:
        raise ValueError(f'ERROR: optimizer {name} is undefined.')
    return optim

class Trainer(object):
    
    def __init__(self, args):

        self.args = args

        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0


        if args.cuda and torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")


        self.Ko = self.args.n_his - (self.args.Kt - 1) * 2 * self.args.stblock_num

        self.args.adj_mat = load_weighted_adjacency_matrix(args.wam_path) #MATRIX!

        n_vertex_vel = pd.read_csv(self.args.data_path, header=None).shape[1]
        n_vertex_adj = pd.read_csv(self.args.wam_path, header=None).shape[1]
        if n_vertex_vel != n_vertex_adj:
            raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
        else:
            self.args.n_vertex = n_vertex_vel
        
        self.args.day_slot = int(24 * 60 / self.args.time_intvl)

        time_pred = self.args.n_pred * self.args.time_intvl
        time_pred_str = str(time_pred) + '_mins'

        self.args.model_save_path = self.args.model_save_path + '_' + self.args.dataset + '_' + time_pred_str + '.pth'
        self.args.blocks = []
        self.args.blocks.append([1])
        for l in range(self.args.stblock_num):
            # blocks.append([64,16,64]) #正常搜索空间
            self.args.blocks.append([64,16,64])
        if self.Ko == 0:
            self.args.blocks.append([128])
        elif self.Ko > 0:
            self.args.blocks.append([128, 128])
        self.args.blocks.append([1])

        self.controller = None

        self.args.all_loss_list = []

        self.build_model() #重要！
        
        controller_optimizer = _get_optimizer(self.args.opt) #选optimizer，默认AdamW

        self.args.loss = nn.MSELoss() 

        self.args.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr, weight_decay=self.args.weight_decay_rate) 

        self.args.scheduler = optim.lr_scheduler.StepLR(self.args.controller_optim, step_size=self.args.step_size, gamma=self.args.gamma)

        self.args.early_stopping = earlystopping.EarlyStopping(patience=2000, path=self.args.model_save_path, verbose=True)

    def build_model(self):

        self.args.format = "micro" #不是默认，用的时候需要改
        self.args.predict_hyper = True #是否要预测超参数


        # 重要部分开始

        self.submodel_manager = MicroCitationManager(self.args) 

        self.search_space = IncrementSearchSpace().get_search_space()

        self.action_list = IncrementSearchSpace().generate_action_list() 

        from STGCNAS_controller import SimpleNASController
        self.controller = SimpleNASController(self.args, action_list=self.action_list, search_space=self.search_space, cuda=self.args.cuda)#新建了一个Agent√
        # 重要部分结束

        if self.cuda:
            self.controller.cuda()


    def form_structure_info(self, temp_actions): #生成内容 √
        actual_action = {} #字典
        actual_action["action"] = temp_actions[:-2]#除了后2个以外前面所有
        actual_action["hyper_param"] = temp_actions[-2:]#后2个
        return actual_action

    def train(self): #Trainer 初始化后开始
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):#start_epoch=0 max_epoch=10

            self.train_controller()

            self.derive(sample_num=self.args.derive_num_sample)

            if self.epoch % self.args.save_epoch == 0: #每 2 次保存一下
                self.save_model()

        if self.args.derive_finally: #上面结束后就训练完了，derive_finally默认为True
            best_actions = self.derive() #选出最好的action！
            print("Best structure:" + str(best_actions))
        self.save_model() #保存

    def train_controller(self): 

        print("*" * 35, "training controller", "*" * 35)

        model = self.controller 

        model.train() #torch的

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size) #初始化 √

        total_loss = 0
        for step in range(self.args.controller_max_step): #controller_max_step强行设为1

            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(with_details=True) 


            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            


            results = self.get_reward(structure_list, np_entropies, hidden) #√
            
            
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = tensor_utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.args.controller_optim.zero_grad() # 梯度清零
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.args.controller_optim.step()

            total_loss += tensor_utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)
        print()


    def get_reward(self, structure_list, entropies, hidden): #√

        if not isinstance(entropies, np.ndarray): #判断一个对象是否是一个已知的类型:entropies是不是np.ndarray
            entropies = entropies.data.cpu().numpy() #转换
        if isinstance(structure_list, dict): #转换
            structure_list = [structure_list]
        if isinstance(structure_list[0], list) or isinstance(structure_list[0], dict):
            pass
        else:
            structure_list = [structure_list]  # when structure_list is one structure

        reward_list = []
        for structures in structure_list:

            structures = self.form_structure_info(structures)
            reward = self.submodel_manager.test_with_param(structures, format=self.args.format)#计算reward用
            
            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[1]

            reward_list.append(reward)

        if self.args.entropy_mode == 'reward': 
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def evaluate(self, temp_actions): #在验证集上验证一个模型的效果 √
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        temp_actions = self.form_structure_info(temp_actions)
        results = self.submodel_manager.retrain(temp_actions, format=self.args.format)
        if results:
            reward, scores = results
        else:
            return

        logger.info(f'eval | {temp_actions} | reward: {reward:8.2f} | scores: {scores:8.2f}')


    def derive_from_history(self):

        with open(self.args.dataset + "_" + self.args.submanager_log_file, "r") as f:
            lines = f.readlines()

        results = []
        best_val_score = "999999"
        for line in lines:
            actions = line[:line.index(";")]
            val_score = line.split(";")[-1]
            results.append((actions, val_score))
        results.sort(key=lambda x: x[-1], reverse=True)
        best_structure = ""
        best_score = 999999
        for actions in results[5:]: #选最好的（后）5个
            actions = eval(actions[0])
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            val_scores_list = []
            for i in range(20):
                val_loss, test_loss = self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_loss)

            tmp_score = np.mean(val_scores_list)
            if tmp_score < best_score:
                best_score = tmp_score
                best_structure = actions

        print("Best structure:" + str(best_structure)) #最好的！
        # train from scratch to get the final score
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        test_scores_list = []
        for i in range(100):
            # manager.shuffle_data()
            val_loss, test_loss = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_loss)
        print(f"Best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure

    def derive(self, sample_num=None):

        # controller_train 类训练好了controller,使用训练好的controller来进行sample
        """
        sample a serial of structures, and return the best structure.
        """
        if sample_num is None and self.args.derive_from_history:
             # 当执行 best_actions = self.derive() 时调用函数derive_from_history()选取最佳action_best
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample #选几个？

            structure_list, _, entropies = self.controller.sample(sample_num, with_details=True)
            # 默认使用训练好的controller采样?个

            max_R = 0
            best_actions = None
            filename = self.model_info_filename

            #对采样的结构进行验证
            for action in structure_list:
                
                structure = self.form_structure_info(action)

                # 测试采样的效果,使用val_score值来评估
                reward = self.submodel_manager.test_with_param(structure, format=self.args.format)

                if reward is None:  # cuda error hanppened
                    continue
                else:
                    results = reward[1] # 获取val_score

                if results > max_R:
                    max_R = results
                    best_actions = action #选择val_score最大的结构

            logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')

            # 验证最佳结构,重新使用数据集训练并得到其val_score与test_score
            self.evaluate(best_actions)

            # 返回最佳结构
            return best_actions



    @property
    def model_info_filename(self):#√
        return f"{self.args.dataset}_{self.args.format}_results.txt"

    @property
    def controller_path(self):#√
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):#√
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self): 
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):

            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        
        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.args.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                tensor_utils.remove_file(path)