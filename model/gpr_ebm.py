import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from .gcn import GCNConv_dgl
from .gin import GINConv_dgl
from .gat import GATConv_dgl


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.2):
        super(MLP, self).__init__()
        # 定义 MLP 层
        self.fc = nn.Linear(in_channels, in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        # nn.init.zeros_(self.fc2.weight)  # 初始化权重为0
        # nn.init.zeros_(self.fc2.bias)    # 初始化偏置为0
        self.dropout_rate = dropout_rate
        
    def forward(self, x, dropout_rate = 0.2, return_x1 = False):
        # MLP 前向传播：使用 ReLU 激活函数和 Dropout
        # x = F.leaky_relu(self.fc(x))
        # x = F.leaky_relu(self.fc1(x))
        # x = F.dropout(x, p=dropout_rate, training=self.training)
        x1 = self.fc2(x)
        if return_x1:
            return F.normalize(x1,dim=1),  F.normalize(x1,dim=1)
        else:
            return F.normalize(x1,dim=1), F.normalize(x1,dim=1)
        


class GPR_EBM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse, num_classes):
        super(GPR_EBM, self).__init__()

        self.inlinear = nn.Linear(in_channels, hidden_channels)
        # 将权重和偏置初始化为零
        # nn.init.zeros_(self.inlinear.weight)  # 初始化权重为0
        # nn.init.zeros_(self.inlinear.bias)    # 初始化偏置为0
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj)
        # self.fc = nn.Linear(hidden_channels, hidden_channels)
        # self.classifier = nn.Linear(hidden_channels, 1)
        self.mlp = MLP(in_channels, in_channels, in_channels)
        # self.mlp = MLP(hidden_channels, hidden_channels, hidden_channels)
        self.proto_mlp = MLP(in_channels, in_channels, in_channels)

        # # 使用均匀分布初始化
        # for param in self.mlp.parameters():
        #     if param.requires_grad:
        #         nn.init.uniform_(param, a=-0.1, b=0.1)

        # # 使用正态分布初始化
        # for param in self.proto_mlp.parameters():
        #     if param.requires_grad:
        #         nn.init.normal_(param, mean=0.0, std=0.02)

        #ETF Initialization
        P = self.generate_random_orthogonal_matrix(in_channels, num_classes,try_assert = True)
        # P = self.generate_random_orthogonal_matrix(hidden_channels, num_classes,try_assert = True)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.T = 1
        self.ori_M = M.cuda()
        self.ori_M.requires_grad_(False)
        # self.LWS = LWS
        # self.reg_ETF = reg_ETF
        # self.BN_H = nn.BatchNorm1d(feat_in)
        # if fix_bn:
        #     self.BN_H.weight.requires_grad = False
        #     self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes,try_assert):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        if try_assert:
            assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P
    
    def inference(self, logits, score='MSP'):
        if score == 'Energy':
            _, pred = torch.max(logits, dim=1)
            score = self.T * torch.logsumexp(logits / self.T, dim=-1)
        elif score == 'MSP':
            sp = torch.softmax(logits, dim=-1)
            score, pred = sp.max(dim=-1)
        return pred, score

    def forward(self, x, g=None, use_h = False):
        x = self.inlinear(x)
        energy, h = self.gnn.forward(x, g)
        # h = self.fc(F.relu(h))
        if use_h:
            return energy, F.normalize(h, dim=1, p=2)
        else:
            return energy
        
    def forward_ETF(self, x, g, dropout_rate=0.0):
        # x = self.BN_H(x)
        # energy, feature = self.forward(x, g,  use_h = True)
        
        feature, x = self.mlp(x,dropout_rate = 0.0)
        # x = x / torch.clamp(
        #     torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return feature
  
    def forward_multi_label(self, x, g, dropout_rate=0.0):
        # x = self.BN_H(x)
        # energy, feature = self.forward(x, g,  use_h = True)
        x = self.inlinear(x)
        h = self.gnn.forward_multi_label(x, g)
        feature, x = self.mlp(h,dropout_rate = 0.0)
        # x = x / torch.clamp(
        #     torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return feature
    
            
    def forward_proto(self, x, g, dropout_rate=0.0):
        # x = self.BN_H(x)
        # energy, feature = self.forward(x, g,  use_h = True)
        feature, x = self.proto_mlp(x,dropout_rate = 0.0)
        # x = x / torch.clamp(
        #     torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return feature


class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl(hidden_channels, hidden_channels) for _ in range(num_layers)])
        # self.layers = nn.ModuleList([GINConv_dgl(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.energy_layers = nn.ModuleList([nn.Linear(hidden_channels, 1) for _ in range(num_layers + 1)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, g=None):
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        energy = self.energy_layers[0](x) * self.temp[0]
        # energy =  torch.logsumexp(x, dim=1) * self.temp[0]
       
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.leaky_relu(x)
            energy += self.energy_layers[i+1](x) * self.temp[i+1]
            # energy += torch.logsumexp(x, dim=1).unsqueeze(1) * self.temp[i+1]
            x = F.dropout(x, p=self.dropout, training=self.training)
        return energy, x
    
    def forward_multi_label(self, x, g=None):
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return  x



class GPRConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GPRConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g, energy):
        with g.local_scope():
            g.ndata['e'] = torch.clamp(energy.clone(), 0, 10)
            g.apply_edges(fn.u_add_v('e', 'e', 'e_e'))
            g.edata['w_new'] = g.edata['w'] * (1.0 / (1 + torch.exp(g.edata['e_e'].squeeze(1))))
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w_new', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']
