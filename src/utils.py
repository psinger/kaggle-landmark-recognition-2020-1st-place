from conf import *

import torch
import random
import numpy as np
import os

from typing import Dict, Tuple, Any

from sklearn.metrics import roc_auc_score
from scipy.special import expit, softmax
from sklearn.metrics import precision_score



def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def global_average_precision_score(y_true, y_pred, ignore_non_landmarks=False):
    indexes = np.argsort(y_pred[1])[::-1]
    queries_with_target = (y_true < args.n_classes).sum()
    correct_predictions = 0
    total_score = 0.
    i = 1
    for k in indexes:
        if ignore_non_landmarks and y_true[k] == args.n_classes:
            continue
        if y_pred[0][k] == args.n_classes:
            continue
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[0][k]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
        i += 1
    return 1 / queries_with_target * total_score

def comp_metric(y_true, logits, ignore_non_landmarks=False):
    
    score = global_average_precision_score(y_true, logits, ignore_non_landmarks=ignore_non_landmarks)
    return score

def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def get_topk_cossim(test_emb, tr_emb, batchsize = 64, k=10, device='cuda:0',verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype = torch.float32, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype = torch.float32, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in test_emb.split(batchsize):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds