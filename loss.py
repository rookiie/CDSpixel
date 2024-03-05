import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

def euclidean(x1,x2): # B,sp,C
	return torch.sum((x1-x2)**2, dim=2).sqrt().mean()

def compute_alignment_loss(content):
    contentA, contentB = content
    b,c,h,w = contentA.shape
    contentA = F.unfold(contentA, kernel_size=16, stride=16).permute(0,2,1).view(b, -1, c, 16 * 16).contiguous() #B, sp, dim, grid
    contentB = F.unfold(contentB, kernel_size=16, stride=16).permute(0,2,1).view(b, -1, c, 16 * 16).contiguous() #B, sp, dim, grid
    
    contentA = torch.mean(contentA, dim=3)
    contentB = torch.mean(contentB, dim=3)
    
    Pa = F.softmax(contentA, dim=2)
    Pb = F.softmax(contentB, dim=2)
    
    kl_divergence = torch.sum(Pa * torch.log(Pa / Pb), dim=2).mean() 
     
    return kl_divergence


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size=8):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)