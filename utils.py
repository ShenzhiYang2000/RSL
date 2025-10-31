# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import dgl
import math
import copy
import time
import dgl.sparse as dglsp
import wandb
import numpy as np
import random as rd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
# import faiss
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import namedtuple
from torch_geometric.data import Data
from pygod.detector import *

EOS = 1e-10


def setup_seed(seed):
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True


class EdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgePredictor, self).__init__()
        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.act = nn.LeakyReLU()
        self.act2 = nn.Tanh()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, x, edge_index):
        x = self.act(self.inlinear(x))
        x = self.act2(self.linear(x))
        res = (1 + self.cos(x[edge_index[0]], x[edge_index[1]])) / 2
        res = torch.clamp(res, 0, 1)
        return res


class EnsembleEdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, n=4):
        super(EnsembleEdgePredictor, self).__init__()
        self.n = n
        self.predictors = nn.ModuleList([EdgePredictor(in_channels, hidden_channels) for _ in range(n)])

    def forward(self, x, edge_index):
        res = [self.predictors[i](x, edge_index) for i in range(self.n)]
        res = torch.mean(torch.stack(res), dim=0)
        return res


def train_edge_predictor(old_g, in_channels, hidden_channels, train_mask, label, device):
    valid_ratio = 0.1
    batch_size = 4096
    epochs = 20

    g = dgl.remove_self_loop(old_g)
    g = g.to(device)
    train_mask = train_mask.to(device)
    label = label.to(device).bool()

    loss_func = nn.BCELoss()

    edge_predictor = EdgePredictor(in_channels, hidden_channels).to(device)
    optimizer = torch.optim.Adam(edge_predictor.parameters(), lr=0.001)

    ind_node_train_mask = (~label) & train_mask

    ind_with_ood_pairs_mask = (label[g.edges()[0]] & (~label[g.edges()[1]])) |\
                              ((~label[g.edges()[0]]) & label[g.edges()[1]])
    ood_node_pairs_mask = label[g.edges()[0]] & label[g.edges()[1]]
    ind_node_pairs_mask = ind_node_train_mask[g.edges()[0]] & ind_node_train_mask[g.edges()[1]]

    ind_node_edges = (g.edges()[0][ind_node_pairs_mask], g.edges()[1][ind_node_pairs_mask])
    ind_node_edge_num = ind_node_edges[0].shape[0]

    ood_node_edges = (g.edges()[0][ood_node_pairs_mask], g.edges()[1][ood_node_pairs_mask])
    ind_with_ood_edges = (g.edges()[0][ind_with_ood_pairs_mask], g.edges()[1][ind_with_ood_pairs_mask])

    neg_node_edges = dgl.sampling.global_uniform_negative_sampling(g, ind_node_edge_num, exclude_self_loops=True)
    rand_idx = torch.randperm(ind_node_edge_num)
    ind_node_edges_shuffle = (ind_node_edges[0][rand_idx], ind_node_edges[1][rand_idx])

    ind_node_edge_train = (ind_node_edges_shuffle[0][:int(ind_node_edge_num * (1 - valid_ratio))], ind_node_edges_shuffle[1][:int(ind_node_edge_num * (1 - valid_ratio))])
    ind_node_edge_valid = (ind_node_edges_shuffle[0][int(ind_node_edge_num * (1 - valid_ratio)):], ind_node_edges_shuffle[1][int(ind_node_edge_num * (1 - valid_ratio)):])
    neg_node_edge_train = (neg_node_edges[0][:int(ind_node_edge_num * (1 - valid_ratio))], neg_node_edges[1][:int(ind_node_edge_num * (1 - valid_ratio))])
    neg_node_edge_valid = (neg_node_edges[0][int(ind_node_edge_num * (1 - valid_ratio)):], neg_node_edges[1][int(ind_node_edge_num * (1 - valid_ratio)):])

    # train
    best_val_loss = 10000
    best_model_weight = None
    for epoch in range(epochs):
        train_loss_epoch = 0

        for i in range(0, ind_node_edge_train[0].shape[0], batch_size):
            pos_edges = (ind_node_edge_train[0][i: i+batch_size], ind_node_edge_train[1][i: i+batch_size])
            neg_edges = (neg_node_edge_train[0][i: i+batch_size], neg_node_edge_train[1][i: i+batch_size])

            edge_predictor.train()
            optimizer.zero_grad()
            pos_res = edge_predictor(g.ndata['h'], pos_edges)
            neg_res = edge_predictor(g.ndata['h'], neg_edges)
            pos_label = torch.ones_like(pos_res)
            neg_label = torch.zeros_like(neg_res)

            train_loss = (loss_func(pos_res, pos_label) + loss_func(neg_res, neg_label)) / 2
            train_loss.backward()
            optimizer.step()
            train_loss_epoch += train_loss.item()

        edge_predictor.eval()
        with torch.no_grad():
            pos_res = edge_predictor(g.ndata['h'], ind_node_edge_valid)
            neg_res = edge_predictor(g.ndata['h'], neg_node_edge_valid)
            pos_label = torch.ones_like(pos_res)
            neg_label = torch.zeros_like(neg_res)
            # print(len(pos_res))
            # print(len(neg_res))
            # pos_res = torch.cat((1 - pos_res, pos_res), dim=1)
            # neg_res = torch.cat((1 - neg_res, neg_res), dim=1)
            val_loss = ((loss_func(pos_res, pos_label) + loss_func(neg_res, neg_label)) / 2).item()

            ood_ood_res = edge_predictor(g.ndata['h'], ood_node_edges).mean()
            ood_ind_res = edge_predictor(g.ndata['h'], ind_with_ood_edges).mean()
            ind_ind_res = edge_predictor(g.ndata['h'], ind_node_edge_valid).mean()

        train_loss_epoch /= (ind_node_edge_train[0].shape[0] // batch_size + 1)
        print(f"train loss: {train_loss_epoch:.4f}, val loss: {val_loss:.4f}, ind_ind_cos: {float(ind_ind_res):.4f}, "
              f"ind_ood_cos: {float(ood_ind_res):.4f}, ood_ood_cos: {float(ood_ood_res):.4f}")
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(edge_predictor.state_dict()).items()}

    edge_predictor.load_state_dict(best_model_weight)
    # ood_ood_res = edge_predictor(g.ndata['h'], ood_node_edges).cpu().detach().numpy()
    # ood_ind_res = edge_predictor(g.ndata['h'], ind_with_ood_edges).cpu().detach().numpy()
    # ind_ind_res = edge_predictor(g.ndata['h'], ind_node_edge_valid).cpu().detach().numpy()
    # np.save('ood_ood.npy', ood_ood_res)
    # np.save('ind_ind.npy', ind_ind_res)
    # np.save('ood_ind.npy', ood_ind_res)
    return edge_predictor


def margin_rank_loss(label, energy):
    loss_func = torch.nn.MarginRankingLoss(margin=1)
    energy_ood = energy[label == 1]
    energy_ind = energy[label == 0]
    new_energy_ood = energy_ood[torch.randint(0, energy_ood.shape[0], (energy_ind.shape[0],))]
    return loss_func(new_energy_ood, energy_ind, torch.ones_like(energy_ind))


def select_pygod_model(model_str, hidden_channels, dropout, lr, weight_decay, batch_size=0, device=0, verbose=True):
    # 'ANOMALOUS', 'AnomalyDAE', 'CoLA', 'CONAD', 'DOMINANT', 'DONE', 'GAAN', 'GCNAE', 'GUIDE', 'MLPAE', 'OCGNN', 'Radar'
    if model_str == 'ANOMALOUS':
        model = ANOMALOUS(lr=lr, gpu=-1, weight_decay=weight_decay, verbose=verbose, epoch=10)

    elif model_str == 'MLPAE':
        model = MLPAE(hid_dim=hidden_channels, dropout=dropout, lr=lr, gpu=device, batch_size=batch_size,
                      weight_decay=weight_decay, verbose=verbose)

    elif model_str == 'GCNAE':
        model = GCNAE(epoch=5, hid_dim=hidden_channels, dropout=dropout, lr=lr, gpu=device, batch_size=batch_size,
                      weight_decay=weight_decay, verbose=verbose)

    elif model_str == 'CONAD':
        model = CONAD(hid_dim=hidden_channels, dropout=dropout, lr=lr, gpu=-1, batch_size=batch_size,
                      weight_decay=weight_decay, verbose=verbose)

    elif model_str == 'CoLA':
        model = CoLA(lr=lr, embedding_dim=hidden_channels, batch_size=batch_size, gpu=device, weight_decay=weight_decay,
                     verbose=verbose)

    elif model_str == 'DOMINANT':
        model = DOMINANT(epoch=20, hid_dim=hidden_channels, dropout=dropout, weight_decay=weight_decay, lr=lr, gpu=0,
                         batch_size=batch_size, verbose=verbose)

    elif model_str == 'GAAN':
        model = GAAN(epoch=20, batch_size=batch_size, hid_dim=hidden_channels, gpu=0, verbose=True, lr=lr, weight_decay=weight_decay)

    else:
        raise NotImplementedError

    return model


def get_sklearn_model(model_str):
    if model_str == 'svm':
        model = sklearn.svm.OneClassSVM()
    elif model_str == 'LocalOutlier':
        model = LocalOutlierFactor()
    elif model_str == 'IsolationForest':
        model = IsolationForest()
    else:
        raise NotImplementedError

    return model


def attack(g, args):
    if args.attack == 'mask_attribute' or 'gaussian':
        return attack_x(g, args)
    elif args.attack == 'fraud_edge_drop':
        return attack_edges(g, args)
    else:
        raise NotImplementedError


def attack_x(g, args):
    print(f'attack node attributes: {args.attack}, ratio: {args.attack_ratio}')
    if args.attack == 'mask_attribute':
        mask = get_random_mask_ogb(g.ndata['feature'], r=args.attack_ratio)
        g.ndata['feature'] = g.ndata['feature'] * (1 - mask)
    elif args.attack == 'gaussian':
        noise = args.attack_ratio ** 0.5 * torch.randn_like(g.ndata['feature'])
        g.ndata['feature'] = g.ndata['feature'] + noise
    return g


def attack_edges(g, args):
    print(f'attack edge: {args.attack}, ratio: {args.attack_ratio}')
    nlabel = g.ndata['label']
    g = dgl.remove_self_loop(g)
    edge_index = g.edges()
    fraud2fraud_edge_label = nlabel[edge_index[0]] * nlabel[edge_index[1]]
    fraud2normal_edge_label = (1 - nlabel[edge_index[0]]) * nlabel[edge_index[1]] + (1 - nlabel[edge_index[1]]) * \
                              nlabel[edge_index[0]]
    normal2normal_edge_label = (1 - nlabel[edge_index[0]]) * (1 - nlabel[edge_index[1]])
    fraud2fraud_edge_index = torch.nonzero(fraud2normal_edge_label)

    fraud2fraud_edges_num = fraud2fraud_edge_index.shape[0]
    removed_fraud2fraud_edge_index = fraud2fraud_edge_index[torch.randperm(
        fraud2fraud_edges_num)[0:int(fraud2fraud_edges_num * args.attack_ratio)]]

    g = dgl.remove_edges(g, removed_fraud2fraud_edge_index.squeeze(1))
    g = dgl.add_self_loop(g)
    return g


def normalize_features(mx, norm_row=True):
    """
    Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """

    if norm_row:
        rowsum = np.array(mx.sum(1)) + 0.01
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    else:
        column_max = mx.max(dim=0)[0].unsqueeze(0)
        column_min = mx.min(dim=0)[0].unsqueeze(0)
        min_max_column_norm = (mx - column_min) / (column_max - column_min)
        # l2_norm = torch.norm(min_max_column_norm, p=2, dim=-1, keepdim=True)
        mx = min_max_column_norm
    return mx


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def get_random_mask(features, r, nr):
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    # pzeros = nones / nzeros / r * nr
    pzeros = nzeros / nones / r * nr
    probs = torch.zeros(features.shape).to(features.device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r
    mask = torch.bernoulli(probs)
    return mask


def get_random_mask_ogb(features, r):
    probs = torch.full(features.shape, r)
    mask = torch.bernoulli(probs)
    return mask


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / \
                              (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            # inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1)))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        # new_matrix = dglsp.val_like(adj, D_value)
        # new_values = adj.values() * D_value
        # return torch.sparse.FloatTensor(adj.indices(), D_value, adj.size()).coalesce()
        return torch.sparse.FloatTensor(adj.indices(), D_value, adj.shape).coalesce()

def gen_dgl_graph(index1, index2, edge_w=None, ndata=None):
    g = dgl.graph((index1, index2))
    if edge_w is not None:
        g.edata['w'] = edge_w
    if ndata is not None:
        g.ndata['h'] = ndata
    return g


def convert_dgl_graph2pyg(g):
    data = Data(x=g.ndata['feature'], edge_index=torch.stack(g.edges()))
    return data


def wandb_summary(test_result):
    wandb.run.summary['best_accuracy'] = test_result.accuracy
    wandb.run.summary['best_recall'] = test_result.recall
    wandb.run.summary['best_macro_f1'] = test_result.macro_F1
    wandb.run.summary['best_auc'] = test_result.auc
    wandb.run.summary['best_gmean'] = test_result.gmean
    wandb.run.summary['best_ap'] = test_result.ap
    wandb.run.summary['best_fpr95'] = test_result.fpr95


def wandb_log(train_loss, val_loss, val_result, epoch):
    wandb.log({"loss": train_loss,
               "Accuracy": val_result.accuracy,
               "Recall": val_result.recall,
               "Macro F1": val_result.macro_F1,
               "AUC": val_result.auc,
               "AP": val_result.ap,
               "Gmean": val_result.gmean,
               "FPR95": val_result.fpr95,
               "Valid Loss": val_loss}, step=epoch)


def print_results(val_results, test_results, record_res = False, args = None, t_type = None):
    valid_aucs, test_aucs = [val_res.auc for val_res in val_results], [
        test_res.auc for test_res in test_results]
    valid_ap, test_ap = [val_res.ap for val_res in val_results], [
        test_res.ap for test_res in test_results]
    valid_macro_f1, test_macro_f1 = [val_res.macro_F1 for val_res in val_results], [
        test_res.macro_F1 for test_res in test_results]
    valid_gmean, test_gmean = [val_res.gmean for val_res in val_results], [
        test_res.gmean for test_res in test_results]
    valid_fpr95, test_fpr95 = [val_res.fpr95 for val_res in val_results], [
        test_res.fpr95 for test_res in test_results]

    print(f"mean+-std of valid auc: {np.mean(valid_aucs):.4f}+-{np.std(valid_aucs):.4f}, test auc: {np.mean(test_aucs):.4f}+-{np.std(test_aucs):.4f}")
    print(f"mean+-std of valid ap: {np.mean(valid_ap):.4f}+-{np.std(valid_ap):.4f}, test ap: {np.mean(test_ap):.4f}+-{np.std(test_ap):.4f}")
    print(f"mean+-std of valid macro f1: {np.mean(valid_macro_f1):.4f}+-{np.std(valid_macro_f1):.4f}, test macro f1: {np.mean(test_macro_f1):.4f}+-{np.std(test_macro_f1):.4f}")
    print(f"mean+-std of valid gmean: {np.mean(valid_gmean):.4f}+-{np.std(valid_gmean):.4f}, test gmean: {np.mean(test_gmean):.4f}+-{np.std(test_gmean):.4f}")
    print(f"mean+-std of valid fpr95: {np.mean(valid_fpr95):.4f}+-{np.std(valid_fpr95):.4f}, test fpr95: {np.mean(test_fpr95):.4f}+-{np.std(test_fpr95):.4f}")
    if record_res:
        import pandas as pd

        # 实验数据
        data = {
            "AUROC": [f'{np.mean(test_aucs):.4f}+-{np.std(test_aucs):.4f}'],
            "AUPR": [f'{np.mean(test_ap):.4f}+-{np.std(test_ap):.4f}'],
            "FPR95": [f'{np.mean(test_fpr95):.4f}+-{np.std(test_fpr95):.4f}']
        }

        # 保存到 CSV 文件（追加模式）
        csv_file_path = f"/data01/ysz/RSL/results/{t_type}/{args.dataset}_experiment_data.csv"

        df = pd.DataFrame(data)
        df.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False)

def print_test_results(test_results):
    test_aucs = [test_res.auc for test_res in test_results]
    test_ap = [test_res.ap for test_res in test_results]
    test_macro_f1 = [test_res.macro_F1 for test_res in test_results]
    test_gmean = [test_res.gmean for test_res in test_results]
    test_fpr95 = [test_res.fpr95 for test_res in test_results]
    print(f"mean+-std of test auc: {np.mean(test_aucs):.4f}+-{np.std(test_aucs):.4f}")
    print(f"mean+-std of test ap: {np.mean(test_ap):.4f}+-{np.std(test_ap):.4f}")
    print(f"mean+-std of test macro f1: {np.mean(test_macro_f1):.4f}+-{np.std(test_macro_f1):.4f}")
    print(f"mean+-std of test gmean: {np.mean(test_gmean):.4f}+-{np.std(test_gmean):.4f}")
    print(f"mean+-std of test fpr95: {np.mean(test_fpr95):.4f}+-{np.std(test_fpr95):.4f}")




# def run_kmeans(x, num_cluster = 10, gpu = 0, temperature = 0.07):
#     """
#     Args:
#         x: data to be clustered
#     """
#     start_time = time.time()
#     print("performing kmeans clustering")
#     results = {"im2cluster": [], "centroids": [], "density": []}

#     # num_cluster = args.num_cluster
#     d = x.shape[1]
#     k = int(num_cluster)
#     clus = faiss.Clustering(d, k)
#     clus.verbose = False
#     clus.niter = 20
#     clus.nredo = 5
#     clus.max_points_per_centroid = 1000
#     clus.min_points_per_centroid = 10

#     res = faiss.StandardGpuResources()
#     cfg = faiss.GpuIndexFlatConfig()
#     cfg.useFloat16 = False
#     cfg.device = gpu
#     index = faiss.GpuIndexFlatL2(res, d, cfg)

#     clus.train(x.cpu().detach(), index)

#     D, I = index.search(x.cpu().detach(), 1)
#     im2cluster = [int(n[0]) for n in I]

#     centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

#     Dcluster = [[] for c in range(k)]
#     for im, i in enumerate(im2cluster):
#         Dcluster[i].append(D[im][0])

#     density = np.zeros(k)
#     for i, dist in enumerate(Dcluster):
#         if len(dist) > 1:
#             d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
#             density[i] = d

#     dmax = density.max()
#     for i, dist in enumerate(Dcluster):
#         if len(dist) <= 1:
#             density[i] = dmax

#     density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
#     density = temperature * density / density.mean()

#     centroids = torch.Tensor(centroids).cuda()
#     centroids = nn.functional.normalize(centroids, p=2, dim=1)

#     im2cluster = torch.LongTensor(im2cluster).cuda()
#     density = torch.Tensor(density).cuda()

#     results["centroids"] = centroids
#     results["density"] = density
#     results["im2cluster"] = im2cluster

#     print("Kmeans end. Eplapsed {} s".format(time.time() - start_time))

#     return results






class ContLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        cont_cutoff=False,
        knn_aug=False,
        num_neighbors=0,
        contrastive_clustering=1,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_clustering = contrastive_clustering
        self.cont_cutoff = cont_cutoff
        self.knn_aug = knn_aug
        self.num_neighbors = num_neighbors

    def forward(self, q, k, cluster_idxes=None, preds=None, start_knn_aug=False):
        batch_size = q.shape[0]

        q_and_k = torch.cat([q, k], dim=0)
        l_i = torch.einsum("nc,kc->nk", [q, q_and_k]) / self.temperature

        # batch_size = 2048  # 根据显存情况选择合适的批量大小
        # l_i = []
        # for i in range(0, q.size(0), batch_size):
        #     q_chunk = q[i:i + batch_size]
        #     l_i_chunk = (q_chunk @ q_and_k.T) / self.temperature
        #     l_i.append(l_i_chunk)
        # l_i = torch.cat(l_i, dim=0)

        self_mask = torch.ones_like(l_i, dtype=torch.float)
        self_mask = (
            torch.scatter(self_mask, 1, torch.arange(batch_size).view(-1, 1).cuda(), 0)
            .detach()
            .cuda()
        )

        positive_mask_i = torch.zeros_like(l_i, dtype=torch.float)
        positive_mask_i = (
            torch.scatter(
                positive_mask_i,
                1,
                batch_size + torch.arange(batch_size).view(-1, 1).cuda(),
                1,
            )
            .detach()
            .cuda()
        )

        l_i_exp = torch.exp(l_i)
        l_i_exp_sum = torch.sum((l_i_exp * self_mask), dim=1, keepdim=True)

        loss = -torch.sum(
            torch.log(l_i_exp / l_i_exp_sum) * positive_mask_i, dim=1
        ).mean()

        if cluster_idxes is not None and self.contrastive_clustering:
            cluster_idxes = cluster_idxes.view(-1, 1)
            cluster_idxes_kq = torch.cat([cluster_idxes, cluster_idxes], dim=0)
            mask = torch.eq(cluster_idxes, cluster_idxes_kq.T).float().cuda()

            if self.cont_cutoff:
                preds = preds.detach()
                pred_labels = (preds > 0.5) * 1
                pred_labels = pred_labels.view(-1, 1)
                pred_labels_kq = torch.cat([pred_labels, pred_labels], dim=0)
                label_mask = torch.eq(pred_labels, pred_labels_kq.T).float().cuda()

                mask = mask * label_mask

            if self.knn_aug and start_knn_aug:
                cosine_corr = q @ q_and_k.T
                _, kNN_index = torch.topk(
                    cosine_corr, k=self.num_neighbors, dim=-1, largest=True
                )
                mask_kNN = torch.scatter(
                    torch.zeros(mask.shape).cuda(), 1, kNN_index, 1
                )
                mask = ((mask + mask_kNN) > 0.5) * 1

            mask = mask.float().detach().cuda()
            batch_size = q.shape[0]
            anchor_dot_contrast = torch.div(
                torch.matmul(q, q_and_k.T), self.temperature
            )
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.scatter(
                torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).cuda(), 0
            )
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            loss_prot = -mean_log_prob_pos.mean()
            loss += loss_prot

        return loss



class BCELoss(nn.Module):
    def __init__(self, ent_loss=False):
        super().__init__()
        self.ent_loss = ent_loss

    def forward(self, preds, label, weight=None):
        preds = torch.sigmoid(preds)
        logits_ = torch.cat([1.0 - preds, preds], dim=1)
        logits_ = torch.clamp(logits_, 1e-4, 1.0 - 1e-4)

        loss_entries = (-label * logits_.log()).sum(dim=0)
        label_num_reverse = 1.0 / label.sum(dim=0)
        label_num_reverse = torch.clamp(label_num_reverse, 1e-4, 1.0 - 1e-4)
        loss = (loss_entries * label_num_reverse).sum()

        if self.ent_loss:
            loss_ent = -(logits_ * logits_.log()).sum(1).mean()
            loss = loss + loss_ent * 0.1
        return loss
    




class EqualNormLoss(nn.Module):
    def __init__(self, pattern='l2') -> None:
        super(EqualNormLoss, self).__init__()
        self.activate = nn.LeakyReLU()
        self.margin = 0.01
        self.pattern = pattern


    def forward(self, features):
        F = features.squeeze(1)
        # F = features.norm(dim=1, p=2)
        # F = features * features
        # F = torch.sum(F, dim = 1)
        F_mean = torch.abs(torch.mean(F)).detach_()
        x = 1 - (F / F_mean)
        if self.pattern == 'l1':
            x = torch.abs(x) 
        elif self.pattern == 'l2':
            x = x * x * F_mean
        ENloss = x.sum() / x.shape[0]

        # modulus = features.norm(dim=1, p=2)
        # modulus_mean = torch.abs(torch.mean(modulus)).detach_()
        # modulus_var = torch.var(modulus)
        # ENloss = modulus_var / (modulus_mean)

        return  ENloss