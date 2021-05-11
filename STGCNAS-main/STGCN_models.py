
import torch
import torch.nn as nn

import layers

from utility import calculate_laplacian_matrix

class STGCN_Conv(nn.Module): #√

    def __init__(self, args, actions, Kt, Ks, blocks, T, n_vertex): 
        super(STGCN_Conv, self).__init__()
        self.args = args
        modules = []
        print("Actions are:")
        print(actions)
        if (actions[1] == "chebconv"):
            if (actions[6] == "rw_rw"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_rw_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = conv_matrix1
            elif (actions[6] == "rw_sym"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_rw_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_sym_normd_lap_mat")).float().to(self.args.device)
            elif (actions[6] == "sym_rw"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_sym_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_rw_normd_lap_mat")).float().to(self.args.device)
            elif (actions[6] == "sym_sym"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "wid_sym_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = conv_matrix1
        else:
            if (actions[6] == "rw_rw"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_rw_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = conv_matrix1
            elif (actions[6] == "rw_sym"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_rw_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_sym_normd_lap_mat")).float().to(self.args.device)
            elif (actions[6] == "sym_rw"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_sym_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_rw_normd_lap_mat")).float().to(self.args.device)
            elif (actions[6] == "sym_sym"):
                conv_matrix1 = torch.from_numpy(calculate_laplacian_matrix(self.args.adj_mat, "hat_sym_normd_lap_mat")).float().to(self.args.device)
                conv_matrix2 = conv_matrix1

        modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[0][-1], blocks[0+1], actions[2], actions[1], conv_matrix1, actions[5], actions[3], actions[4]))
        modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[1][-1], blocks[1+1], actions[2], actions[1], conv_matrix2, actions[5], actions[3], actions[4]))
        self.st_blocks = nn.Sequential(*modules)

        Ko = T - 2 * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, actions[2], actions[5], actions[3])
        elif self.Ko == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.softsign = nn.Softsign()
            self.relu = nn.ReLU()
            self.softplus = nn.Softplus()
            self.leakyrelu = nn.LeakyReLU()
            self.prelu = nn.PReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'softsign':
                x_act_func = self.softsign(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'softplus':
                x_act_func = self.softplus(x_fc1)
            elif self.act_func == 'leakyrelu':
                x_act_func = self.leakyrelu(x_fc1)
            elif self.act_func == 'prelu':
                x_act_func = self.prelu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            x_out = x_fc2
        return x_out
