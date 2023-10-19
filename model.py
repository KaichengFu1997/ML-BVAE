from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_inputs,n_outputs,n_latents,H):
        super(MVAE, self).__init__()
        self.left_encoder = BrainEncoder(n_inputs=n_inputs,n_latents=n_latents)
        self.left_decoder = BrainDecoder(n_outputs=n_outputs,n_latents=n_latents)
        self.right_encoder  = BrainEncoder(n_inputs=n_inputs,n_latents=n_latents)
        self.right_decoder  = BrainDecoder(n_outputs=n_outputs,n_latents=n_latents)
        self.diff_encoder = BrainEncoder(n_inputs=n_inputs,n_latents=n_latents)
        self.diff_decoder = BrainDecoder(n_outputs=n_outputs,n_latents=n_latents)
        self.experts       = ProductOfExperts()
        self.n_latents     = n_latents
        self.mlp = Model1(H)
        # self.mlp = nn.Linear(64,27)
    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, left=None, right=None,diff=None):
        mu, logvar = self.infer(left, right,diff)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        y = self.mlp(z)
        left_recon  = self.left_decoder(z)
        right_recon  = self.right_decoder(z)
        diff_recon = self.diff_decoder(z)
        return left_recon, right_recon,diff_recon, mu, logvar,y

    def infer(self, left=None, right=None,diff=None):
        if left is not None:
            batch_size = left.size(0)
        elif right is not None:
            batch_size = right.size(0)
        else:
            batch_size = diff.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if left is not None:
            left_mu, left_logvar = self.left_encoder(left)
            mu     = torch.cat((mu, left_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, left_logvar.unsqueeze(0)), dim=0)

        if right is not None:
            right_mu, right_logvar = self.right_encoder(right)
            mu     = torch.cat((mu, right_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, right_logvar.unsqueeze(0)), dim=0)

        if diff is not None:
            diff_mu, diff_logvar = self.diff_encoder(diff)
            mu     = torch.cat((mu, diff_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, diff_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class BrainEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_inputs,n_latents):
        super(BrainEncoder, self).__init__()
        self.fc1   = nn.Linear(n_inputs, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc31  = nn.Linear(512, n_latents)
        self.fc32  = nn.Linear(512, n_latents)
        self.swish = nn.ReLU()
    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class BrainDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_outputs,n_latents):
        super(BrainDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512,n_outputs)
        self.swish = nn.ReLU()


    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.fc3(h)
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py




class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class Model1(nn.Module):
    def __init__(self,H):
        super(Model1, self).__init__()
        self.classifier1 = nn.Linear(64,27)
        self.classifier2 = nn.Linear(64,1)
        self.attention_layer = SelfAttention(dim = 64, heads= 1,mask = H)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        # self.feature_extract = torch.nn.Linear(input_size, 64)
    def forward(self, x):
        # x = self.feature_extract(x)
        ya = self.classifier1(x)
        label_embedding = x.unsqueeze(2).repeat(1, 1, 27).permute(0, 2, 1)
        label_embedding = label_embedding * self.classifier1.weight
        label_embedding_refined = self.attention_layer(label_embedding)
        yb = self.classifier2(label_embedding_refined).squeeze(2)
        # alpha = F.sigmoid(self.alpha)
        y =  ya + yb
        return y

class SelfAttention(nn.Module):
    def __init__(
        self, dim,mask=0, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0,
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.mask = mask
    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn + self.mask).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
