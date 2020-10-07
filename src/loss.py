from conf import *

import torch
from torch import nn
import math


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        #print(self.gamma)
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean"):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        
        if crit == "focal":
            self.crit = FocalLoss(gamma=args.focal_loss_gamma)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if args.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if args.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def loss_fn(metric_crit, target_dict, output_dict, val=False):
    
    y_true = target_dict['target']
    y_pred = output_dict['logits']
    #ignore invalid classes for val loss
    mask = y_true < args.n_classes
    if mask.sum() == 0:
        return torch.zeros(1,  device = y_pred.device)
    loss = metric_crit(y_pred[mask], y_true[mask])

    return loss

