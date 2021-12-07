import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparams import is_CUDA



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dense1 = nn.Linear(768, 256)
        self.activation1 = nn.ReLU()
        self.dense2 = nn.Linear(256, 64)
        self.activation2 = nn.ReLU()
        # construct mu and sigma func
        self.dense_x1 = nn.Linear(64, 16)
        self.activation_x = nn.ReLU()
        self.dense_x2 = nn.Linear(16, 1)
        self.dense_s1 = nn.Linear(64, 16)
        self.activation_s = nn.ReLU()
        self.dense_s2 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.activation2(self.dense2(self.activation1(self.dense1(x))))
        # mu
        mu = self.dense_x2(self.activation_x(self.dense_x1(out)))
        logsigma = self.dense_s2(self.activation_s(self.dense_s1(out)))
        return mu, logsigma

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(128, 256)
        self.activation1 = nn.ReLU()
        self.dense2 = nn.Linear(256, 768)
    def forward(self, z):
        return self.dense2(self.activation1(self.dense1(z)))


class ProteinVAEVector(nn.Module):
    def __init__(self):
        super(ProteinVAEVector, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        # encode
        batch_size = x.shape[0]
        mu, logsigma = self.encoder(x)
        # sampling through mu and log_sigma
        if is_CUDA == True:
            mu_s = mu.cpu().view(-1).detach().numpy().tolist() # now mu is a list with len = batch_size
            sigma_s = torch.exp(logsigma).cpu().view(-1).detach().numpy().tolist()
        z = torch.zeros(0, 128)
        for i in range(0, batch_size):
            mu_now = mu_s[i]
            sigma_now = sigma_s[i]
            z_0_now = torch.normal(mu_now, sigma_now, size=(1, 128)) # batch_size, num of features
            z = torch.cat((z,sigma_now*z_0_now+mu_now), dim = 0)
        if is_CUDA == True:
            z = z.cuda()
        # decode
        x_out = self.decoder(z)
        return x_out, mu, logsigma


# ELBO loss
def ELBO(x, x_out, mu, logsigma):
    ELBO_CE = torch.sum((x-x_out)**2)/(x_out.shape[0]*x_out.shape[1])
    batch_size = x.shape[0]
    ELBO_KL = -0.5*(1.0*batch_size + torch.sum(logsigma) - torch.sum(mu**2) - torch.sum(torch.exp(logsigma)))/batch_size
    return ELBO_CE, ELBO_KL
