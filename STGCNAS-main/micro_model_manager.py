from pyg_structure_model_manager import GeoCitationManager 
from STGCN_models import STGCN_Conv
import torch
import os.path as osp
import torch_geometric.transforms as T


class MicroCitationManager(GeoCitationManager):

    def __init__(self, args):
        # print("Olivia5")
        super(MicroCitationManager, self).__init__(args) #维护了一个data

    def build_gnn(self, all_action): #√
        self.Ks = 3
        if (all_action['action'][1] == 'gcnconv'): #X
            self.Ks = 2 #不带matrix，在STGCN_Conv内由actions[1]来调整

        model = STGCN_Conv(self.args, all_action['action'], self.args.Kt, self.Ks, self.args.blocks, self.args.n_his, self.args.n_vertex).to(self.args.device) 

        return model

    def train(self, actions=None, format="micro"):#√
        self.current_action = actions 
        all_action = actions
        param = all_action['hyper_param']
        self.args.lr = param[0]
        self.args.weight_decay = param[1]
        return super(GeoCitationManager, self).train(all_action, format=format)
    #√

    def evaluate(self, all_action=None, format="micro"): #√ actions
        print("Actions is here :")
        print(all_action)
        model_actions = all_action["action"]
        param = all_action["hyper_param"]
        self.args.lr = param[0]
        self.args.weight_decay = param[1]
        return super(GeoCitationManager, self).evaluate(all_action, format=format)
