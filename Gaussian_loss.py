import json
import math
import os

import numpy
import torch
import torch.nn as nn
import sklearn.datasets
from scipy import stats
import numpy as np
import pandas as pd
from torch.autograd import Variable
class GaussianLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=64, feat_dim=640, use_gpu=True):
        super(GaussianLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.martix_size = self.feat_dim * self.num_classes
        if self.use_gpu:
            self.centers = nn.Parameter(torch.normal(mean=0.3,std=0.1,size=(self.num_classes, self.feat_dim)).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        with torch.no_grad():
            self.centers.data = self.centers.clamp(min=1e-1).data
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # labels  => (128,10)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))  # mask => (128,10)

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

