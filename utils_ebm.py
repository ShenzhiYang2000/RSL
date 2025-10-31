import math
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    precision_recall_curve, roc_curve, auc
from collections import namedtuple
from pygod.metric import eval_precision_at_k
from sklearn.utils import assert_all_finite

EPS = 1e-5


class Langevin_Sampler(object):
    def __init__(self, in_dim, args, edge_predictor=None, ood_samples=None, update=False):
        self.in_dim = in_dim
        self.k = ood_samples.shape[0]
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.nsteps = args.nsteps
        self.device = args.device
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        self.buffer_rate = args.buffer_rate
        self.update = update
        self.edge_predictor = edge_predictor

        self.replay_buffer = self._init_replay_buffer(ood_samples)

    def _init_replay_buffer(self, ood_samples):
        assert self.k <= self.buffer_size
        buffer = torch.FloatTensor(self.buffer_size, self.in_dim).uniform_(-1, 1)
        if self.k > 0:
            buffer[:self.k] = ood_samples
        return buffer

    def _init_random(self, n):
        return torch.FloatTensor(n, self.in_dim).uniform_(-1, 1)

    def sample_p_0(self):
        if self.k == 0:
            inds = torch.randint(0, self.buffer_size, (self.batch_size,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size)
            buffer_or_random = (torch.rand(self.batch_size) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
        else:
            assert self.batch_size > self.k
            inds = torch.randint(self.k, self.buffer_size, (self.batch_size - self.k,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size - self.k)
            buffer_or_random = (torch.rand(self.batch_size - self.k) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
            sample_0 = torch.cat((sample_0, self.replay_buffer[:self.k]), dim=0)
        return sample_0, inds

    def sample_q(self, model, replace_inds, features, g):
        model.eval()
        init_sample, buffer_inds = self.sample_p_0()
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)

        # langevin dynamics
        for i in range(self.nsteps):
            replaced_features = self.generate_q_features(x_k, features, replace_inds)
            mean_energy = model(replaced_features, g)[replace_inds].mean()
            # print(f'mean_energy: {mean_energy:.4f}')
            f_prime = torch.autograd.grad(mean_energy, [x_k], retain_graph=True)[0]
            x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)

        model.train()
        final_samples = x_k.detach()
        # update replay buffer
        # not update ground truth fraud samples
        if self.update:
            update_inds = buffer_inds[buffer_inds >= self.k]
            self.replay_buffer[update_inds] = final_samples.cpu()[torch.randperm(self.batch_size)[:update_inds.shape[0]]]
        return self.generate_q_features(final_samples, features, replace_inds)

    def generate_q_features(self, q_features, features, replace_inds):
        replaced_features = features.clone()
        replaced_features[replace_inds] = q_features
        return replaced_features









class Langevin_Sampler1(object):
    def __init__(self, in_dim, args, edge_predictor=None, ood_samples=None, update=False, target_vector=None):
        self.in_dim = in_dim
        self.k = ood_samples.shape[0]
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.nsteps = args.nsteps
        self.device = args.device
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        self.buffer_rate = args.buffer_rate
        self.update = update
        self.edge_predictor = edge_predictor
        self.target_vector = target_vector.to(self.device) if target_vector is not None else None

        self.replay_buffer = self._init_replay_buffer(ood_samples)

    def _init_replay_buffer(self, ood_samples):
        assert self.k <= self.buffer_size
        buffer = torch.FloatTensor(self.buffer_size, self.in_dim).uniform_(-1, 1)
        if self.k > 0:
            buffer[:self.k] = ood_samples
        return buffer

    def _init_random(self, n):
        return torch.FloatTensor(n, self.in_dim).uniform_(-1, 1)

    def sample_p_0(self):
        if self.k == 0:
            inds = torch.randint(0, self.buffer_size, (self.batch_size,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size)
            buffer_or_random = (torch.rand(self.batch_size) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
        else:
            assert self.batch_size > self.k
            inds = torch.randint(self.k, self.buffer_size, (self.batch_size - self.k,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size - self.k)
            buffer_or_random = (torch.rand(self.batch_size - self.k) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
            sample_0 = torch.cat((sample_0, self.replay_buffer[:self.k]), dim=0)
        return sample_0, inds

    def sample_q(self, model, replace_inds, features, g):
        model.eval()
        init_sample, buffer_inds = self.sample_p_0()
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)

        # Langevin dynamics with guidance towards target_vector
        for i in range(self.nsteps):
            replaced_features = self.generate_q_features(x_k, features, replace_inds)
            mean_energy = model(replaced_features, g)[replace_inds].mean()
            f_prime = torch.autograd.grad(mean_energy, [x_k], retain_graph=True)[0]

            # Update with guidance towards target_vector
            if self.target_vector is not None:
                guidance = self.target_vector - x_k  # Guidance term
                x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k) + 0.1 * guidance  # Adjust the scale of guidance

            else:
                x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)

        model.train()
        final_samples = x_k.detach()
        # Update replay buffer
        if self.update:
            update_inds = buffer_inds[buffer_inds >= self.k]
            self.replay_buffer[update_inds] = final_samples.cpu()[torch.randperm(self.batch_size)[:update_inds.shape[0]]]
        return self.generate_q_features(final_samples, features, replace_inds)

    def generate_q_features(self, q_features, features, replace_inds):
        replaced_features = features.clone()
        replaced_features[replace_inds] = q_features
        return replaced_features



class Langevin_Sampler3(object):
    def __init__(self, in_dim, args, edge_predictor=None, ood_samples=None, update=False, target_vector=None):
        self.in_dim = in_dim
        self.k = ood_samples.shape[0]
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.nsteps = args.nsteps
        self.device = args.device
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        self.buffer_rate = args.buffer_rate
        self.update = update
        self.edge_predictor = edge_predictor
        self.target_vector = target_vector.to(self.device) if target_vector is not None else None

        self.replay_buffer = self._init_replay_buffer(ood_samples)

    def _init_replay_buffer(self, ood_samples):
        assert self.k <= self.buffer_size
        buffer = torch.FloatTensor(self.buffer_size, self.in_dim).uniform_(-1, 1)
        if self.k > 0:
            buffer[:self.k] = ood_samples
        return buffer

    def _init_random(self, n):
        return torch.FloatTensor(n, self.in_dim).uniform_(-1, 1)

    def sample_p_0(self):
        if self.k == 0:
            inds = torch.randint(0, self.buffer_size, (self.batch_size,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size)
            buffer_or_random = (torch.rand(self.batch_size) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
        else:
            assert self.batch_size > self.k
            inds = torch.randint(self.k, self.buffer_size, (self.batch_size - self.k,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size - self.k)
            buffer_or_random = (torch.rand(self.batch_size - self.k) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
            sample_0 = torch.cat((sample_0, self.replay_buffer[:self.k]), dim=0)
        return sample_0, inds

    def sample_q(self, model, replace_inds, features, g):
        model.eval()
        init_sample, buffer_inds = self.sample_p_0()
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)

        # Langevin dynamics with guidance towards target_vector
        for i in range(self.nsteps):
            replaced_features = self.generate_q_features(x_k, features, replace_inds)
            mean_energy = model(replaced_features, g)[replace_inds].mean()
            f_prime = torch.autograd.grad(mean_energy, [x_k], retain_graph=True)[0]

            # Update with guidance towards target_vector
            if self.target_vector is not None:
                guidance = self.target_vector - x_k  # Guidance term
                # x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k) + 0.1 * guidance  # Adjust the scale of guidance
                x_k.data = 2 * self.alpha * (x_k.data -  self.sgld_lr * f_prime +  self.sgld_std * torch.randn_like(x_k) ) + 2 * (1-self.alpha) * guidance  # squirrel
                # x_k.data = x_k.data + 1 * guidance  #  All except squirrel
            else:
                x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)

        model.train()
        final_samples = x_k.detach()
        # Update replay buffer
        if self.update:
            update_inds = buffer_inds[buffer_inds >= self.k]
            self.replay_buffer[update_inds] = final_samples.cpu()[torch.randperm(self.batch_size)[:update_inds.shape[0]]]
        return self.generate_q_features(final_samples, features, replace_inds)

    def generate_q_features(self, q_features, features, replace_inds):
        replaced_features = features.clone()
        replaced_features[replace_inds] = q_features
        return replaced_features



class Langevin_Sampler2(object):
    def __init__(self, in_dim, args, edge_predictor=None, ood_samples=None, update=False, target_vectors=None):
        self.in_dim = in_dim
        self.k = ood_samples.shape[0]
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.nsteps = args.nsteps
        self.device = args.device
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        self.buffer_rate = args.buffer_rate
        self.alpha = args.alpha
        self.update = update
        self.edge_predictor = edge_predictor
        # self.target_vectors = [tv.to(self.device) for tv in target_vectors] if target_vectors is not None else None
        self.target_vectors = [tv.to(self.device) for tv in target_vectors] if target_vectors is not None else None

        self.replay_buffer = self._init_replay_buffer(ood_samples)

    def _init_replay_buffer(self, ood_samples):
        assert self.k <= self.buffer_size
        buffer = torch.FloatTensor(self.buffer_size, self.in_dim).uniform_(-1, 1)
        if self.k > 0:
            buffer[:self.k] = ood_samples
        return buffer

    def _init_random(self, n):
        # return torch.FloatTensor(n, self.in_dim).uniform_(-1, 1)
        return torch.zeros(n, self.in_dim)

    def sample_p_0(self):
        if self.k == 0:
            inds = torch.randint(0, self.buffer_size, (self.batch_size,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size)
            buffer_or_random = (torch.rand(self.batch_size) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
        else:
            assert self.batch_size > self.k
            inds = torch.randint(self.k, self.buffer_size, (self.batch_size - self.k,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size - self.k)
            buffer_or_random = (torch.rand(self.batch_size - self.k) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
            sample_0 = torch.cat((sample_0, self.replay_buffer[:self.k]), dim=0)
        return sample_0, inds

    def sample_q(self, model, replace_inds, features, g):
        model.eval()
        init_sample, buffer_inds = self.sample_p_0()
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)

        # Langevin dynamics with guidance towards multiple target vectors
        gap = []
        for i in range(self.nsteps):
            replaced_features = self.generate_q_features(x_k, features, replace_inds)
            mean_energy = model(replaced_features, g)[replace_inds].mean()
            f_prime = torch.autograd.grad(mean_energy, [x_k], retain_graph=True)[0]

            # Update with guidance towards target_vectors
            if self.target_vectors is not None:
                guidance = torch.zeros_like(x_k)
                for target_vector in self.target_vectors:
                    guidance += (target_vector - x_k)  # Accumulate guidance from all target vectors
                guidance /= len(self.target_vectors)  # Average guidance
                gap.append(guidance.sum().item())

                x_k.data = 2 * self.alpha * (x_k.data -  self.sgld_lr * f_prime +  self.sgld_std * torch.randn_like(x_k) ) + 2 * (1-self.alpha) * guidance  # squirrel
                # x_k.data = x_k.data + 1 * guidance  #  All except squirrel

            else:
                x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)
        if len(gap) != 0:
            # print("gap between target_vector and replaced features:")
            # print('gap: ','{:.2f}'.format(gap[0]),'{:.2f}'.format(gap[len(gap)//2]),'{:.2f}'.format(gap[-1]))
            pass
        model.train()
        final_samples = x_k.detach()
        # Update replay buffer
        if self.update:
            update_inds = buffer_inds[buffer_inds >= self.k]
            self.replay_buffer[update_inds] = final_samples.cpu()[torch.randperm(self.batch_size)[:update_inds.shape[0]]]
        return self.generate_q_features(final_samples, features, replace_inds)

    def generate_q_features(self, q_features, features, replace_inds):
        replaced_features = features.clone()
        replaced_features[replace_inds] = q_features
        return replaced_features






def grad_out(model, mask, features, adj, g):
    model.eval()
    x = torch.autograd.Variable(features, requires_grad=True)
    mean_energy = model(x, adj, g)[mask].mean()
    f_prime = torch.autograd.grad(mean_energy, [x], retain_graph=True)[0]

    grad_norm = torch.norm(f_prime, p=2, dim=-1)
    model.train()
    return grad_norm


def best_f1_value_threshold(preds, labels):
    assert_all_finite(labels)
    assert_all_finite(preds)
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return thresholds[best_f1_score_index]


Evaluation_Metrics = namedtuple('Evaluation_Metrics', ['accuracy',
                                                       'macro_F1',
                                                       'recall',
                                                       'auc',
                                                       'ap',
                                                       'gmean',
                                                       'fpr95'])


def fpr95_score(pred, label):
    # calculate false positive rate (OOD -> ID) at 95% true negative rate(ID->ID)
    pred = 1 - pred
    label = 1 - label
    fpr, tpr, thresh = roc_curve(label, pred)
    return fpr[np.where(tpr > 0.95)[0][0]]



def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level= 0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]



# def get_measures(_pos, _neg, recall_level=0.95):
#     pos = np.array(_pos[:]).reshape((-1, 1))
#     neg = np.array(_neg[:]).reshape((-1, 1))
#     examples = np.squeeze(np.vstack((pos, neg)))
#     labels = np.zeros(len(examples), dtype=np.int32)
#     labels[:len(pos)] += 1

#     auroc = roc_auc_score(labels, examples)
#     aupr = average_precision_score(labels, examples)
#     fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

#     return auroc, aupr, fpr, threshould


def get_measures(energy, labels, recall_level=0.95):
    # pos = energy[labels==0]
    # neg = energy[labels==1]

    # examples = np.squeeze(np.vstack((pos, neg)))
    # labels = np.zeros(len(examples), dtype=np.int32)
    # labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, energy)
    aupr = average_precision_score(labels, energy)
    fpr, threshould = fpr_and_fdr_at_recall(labels, energy, recall_level)

    # return auroc, aupr, fpr, threshould
    energy_norm = (energy - min(energy)) / (max(energy) - min(energy) + EPS)

    if threshold is None:
        threshold = best_f1_value_threshold(energy_norm, labels)

    pred_label = np.where(energy_norm > threshould, 1, 0)
    accuracy = accuracy_score(labels, pred_label)
    f1 = f1_score(labels, pred_label, average='macro')
    recall = recall_score(labels, pred_label, average='macro')
    gmean_value = gmean(labels, pred_label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auroc, ap=aupr, gmean=gmean_value,
                              fpr95=fpr), threshould



# def evaluation_ebm_prediction(energy, label, threshold=None):
#     # threshold are chosen in training set !!

#     energy_norm = (energy - min(energy)) / (max(energy) - min(energy) + EPS)
#     # energy_norm = energy
#     # if threshold is None:
#     # threshold = best_f1_value_threshold(energy_norm, label)
#     threshold = best_f1_value_threshold(energy_norm, label)

#     pred_label = np.where(energy_norm > threshold, 1, 0)

#     accuracy = accuracy_score(label, pred_label)
#     f1 = f1_score(label, pred_label, average='macro')
#     recall = recall_score(label, pred_label, average='macro')
#     # auc_value = roc_auc_score(label, energy_norm)
#     # precision_aupr, recall_aupr, _ = precision_recall_curve(label, energy_norm)
#     # ap = auc(recall_aupr, precision_aupr)
#     auc_value = roc_auc_score(label, energy)
#     ap = average_precision_score(label, energy)
#     # ap = average_precision_score(label, energy_norm, average='macro')
#     gmean_value = gmean(label, pred_label)
#     fpr95 = fpr95_score(energy_norm, label)

#     return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc_value, ap=ap, gmean=gmean_value,
#                               fpr95=fpr95), threshold




def evaluation_ebm_prediction(energy, label, threshold=None):
    # threshold are chosen in training set !!

    energy_norm = (energy - min(energy)) / (max(energy) - min(energy) + EPS)
    # energy_norm = energy
    # if threshold is None:
    # threshold = best_f1_value_threshold(energy_norm, label)
    threshold = best_f1_value_threshold(energy_norm, label)

    pred_label = np.where(energy_norm > threshold, 1, 0)

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    # auc_value = roc_auc_score(label, energy_norm)
    # precision_aupr, recall_aupr, _ = precision_recall_curve(label, energy_norm)
    # ap = auc(recall_aupr, precision_aupr)
    auc_value = roc_auc_score(label, energy)
    ap = average_precision_score(label, energy)
    # ap = average_precision_score(label, energy_norm, average='macro')
    gmean_value = gmean(label, pred_label)
    fpr95 = fpr95_score(energy_norm, label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc_value, ap=ap, gmean=gmean_value,
                              fpr95=fpr95), threshold




def evaluation_model_prediction(pred_logit, label):
    pred_label = np.argmax(pred_logit, axis=1)
    pred_logit = pred_logit[:, 1]

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    auc_value = roc_auc_score(label, pred_logit)
    precision_aupr, recall_aupr, _ = precision_recall_curve(label, pred_logit)
    ap = auc(recall_aupr, precision_aupr)
    gmean_value = gmean(label, pred_label)
    fpr95 = fpr95_score(pred_logit, label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc_value, ap=ap, gmean=gmean_value,
                              fpr95=fpr95)


def gmean(y_true, y_pred):
    """binary geometric mean of  True Positive Rate (TPR) and True Negative Rate (TNR)

    Args:
            y_true (np.array): label
            y_pred (np.array): prediction
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    for sample_true, sample_pred in zip(y_true, y_pred):
        TP += sample_true * sample_pred
        TN += (1 - sample_true) * (1 - sample_pred)
        FP += (1 - sample_true) * sample_pred
        FN += sample_true * (1 - sample_pred)

    return math.sqrt(TP * TN / (TP + FN) / (TN + FP))
