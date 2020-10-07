from conf import *
from utils import *

#from pytorchcv.model_provider import get_model as ptcv_get_model
import timm
from torch import nn

import math
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    
class Backbone(nn.Module):

    
    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)
        
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name: #works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x

    
class Net(nn.Module):
    def __init__(self, args, pretrained=True):
        super(Net, self).__init__()
        
        self.args = args
        self.backbone = Backbone(args.backbone, pretrained=pretrained)
        
        if args.pool == "gem":
            self.global_pool = GeM(p_trainable=args.p_trainable)
        elif args.pool == "identity":
            self.global_pool = torch.nn.Identity()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_size = args.embedding_size        
        
        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if args.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif args.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        self.head = ArcMarginProduct(self.embedding_size, args.n_classes)
        
        if args.pretrained_weights is not None:
            self.load_state_dict(torch.load(args.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from',args.pretrained_weights)

    def forward(self, input_dict, get_embeddings=False, get_attentions=False):

        x = input_dict['input']
        x = self.backbone(x)
        
        x = self.global_pool(x)
        x = x[:,:,0,0]
        
        x = self.neck(x)

        logits = self.head(x)
        
        if get_embeddings:
            return {'logits': logits, 'embeddings': x}
        else:
            return {'logits': logits}