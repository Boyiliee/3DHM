from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.autograd import Variable


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        
        self.reconstruction_loss = nn.MSELoss(reduction='mean')

    def forward(self, num_classes, images, labels, X_l2norm, reconstructions, 
                lambda_value=0.5, margin=0.1, scale_factor=0.0005):
                
        self.num_classes = num_classes

        left = F.relu((1-margin) - X_l2norm)
        right = F.relu(X_l2norm - margin)
        loss = labels * left + lambda_value * (1.0 - labels) * right
        margin_loss = loss.sum(dim=1).mean()
        
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return margin_loss + scale_factor * reconstruction_loss
    
    
class MarginLoss(nn.Module):
    """Margin loss for capsulenet"""
    def __init__(self):
        super(MarginLoss, self).__init__()
        
    def forward(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss
    
class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

