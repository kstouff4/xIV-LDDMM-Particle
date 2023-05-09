import os
import time
import numpy as np
from numpy import random
import scipy as sp

import torch
from torch.autograd import grad

import pykeops
from pykeops.torch import Vi, Vj

np_dtype = "float32" #"float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

from matplotlib import pyplot as plt
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
    
import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf
sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init

#####################################################################################
# Varifold Norms

# <\mu_S,\mu_T> = \sum_ij k(x^s_i - x^t_j)<f^s_i,f^t_j>
def GaussLinKernelSingle(sig,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    K = (-D2 / (2.0*sig*sig)).exp() * (u * v).sum()
    return K.sum_reduction(axis=1)

# \sum_sigma \beta/2 <\mu_S,\mu_T>
def GaussLinKernel(sigma,d,l,beta):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        K = (-D2 / (2.0*sig*sig)).exp() * (u * v).sum()
        if sInd == 0:
            retVal = beta[sInd]*K 
        else:
            retVal += beta[sInd]*K
    return (retVal).sum_reduction(axis=1)

########################################################################################
# Apply Transformation

def applyRigid2D(S,nu_S,theta,tau):
    A = torch.zeros((2,2)).type(dtype)
    A[0,0] = torch.cos(theta).type(dtype)
    A[1,1] = torch.cos(theta).type(dtype)
    A[0,1] = -torch.sin(theta).type(dtype)
    A[1,0] = torch.sin(theta).type(dtype)
    
    xc = torch.mean(torch.sum(nu_S,axis=-1)*S,axis=0)
    
    Snew = (S - sc)@A.T + tau
    return Snew

def getSlicesFromWhole(files,zU):
    ## assume files need to be split by z with Z indices taken from original (list of Z's)
    info = np.load(files)
    X = info[info.files[0]]
    nu_X = info[info.files[1]]
        
    zs = np.asarray(zU)
    D = (X[...,-1][...,None] - zs[None,...])**2
    slIndex = np.argmin(D,axis=-1)
    
    Slist = []
    nu_Slist = []
    for i in range(len(zU)):
        Slist.append(torch.tensor(X[slIndex == i,0:2]).type(dtype)) # ignore z 
        nu_Slist.append(torch.tensor(nu_X[slIndex == i,...]).type(dtype))
    return Slist, nu_Slist, slIndex 

def align(Slist,nu_Slist,sigma,its,savedir):
    
    K = GaussLinKernelSingle(sigma,2,nu_Slist[0].shape[-1])
    p0 = torch.zeros((len(Slist)-2)*3).requires_grad_(True).type(dtype)
    lossList = []
    
    def make_loss(Slist,nu_Slist):
        
        def loss(p0):
            L = torch.tensor(0.0).type(dtype)
            for i in range(len(Slist)-2):
                theta = p0[i*3]
                tau = p0[1+i*3:(i+1)*3]
                if (i == 0):
                    L += K(Slist[i],applyRigid2D(Slist[i+1],nu_Slist[i+1],theta,tau),nu_Slist[i],nu_Slist[i+1]).sum()
                else:
                    L += K(applyRigid2D(Slist[i],nu_Slist[i],p0[3*(i-1)],p0[1+(i-1)*3:i*3]),applyRigid2D(Slist[i+1],nu_Slist[i+1],theta,tau),nu_Slist[i],nu_Slist[i+1]).sum()
            L += K(applyrigid2D(Slist[-2],nu_Slist[-2],p0[-3],p0[-2:]),Slist[-1],nu_Slist[-2],nu_Slist[-1]).sum() 
            return L
        return loss
            
    def closure():
        optimizer.zero_grad()
        L = loss(p0)
        print("loss, ", L.detach().cpu().numpy())
        lossList.append(np.copy(L.detach().cpu().numpy()))
        L.backward()
        return L
    
    loss = make_loss(Slist,nu_Slist)
    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    for i in range(its):
        optimizer.step(closure)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossList)),np.asarray(lossList),label="Total Cost, Final = {0:.6f}".format(lossList[-1]))
    ax.legend()
    f.savefig(savedir + 'cost.png',dpi=300)
    
    Snew = [Slist[0]]
    thetas = [0]
    taus = [np.zeros(2)]
    for i in range(len(Slist)-1):
        Snew.append(applyRigid2D(Slist[i+1],nu_Slist[i+1],p0[i*3],p0[1+i*3:(i+1)*3]))
        thetas.append(p0[i*3].detach().cpu().numpy())
        taus.append(p0[1+i*3:(i+1)*3].detach().cpu().numpy())
    Snew.append(Slist[-1])
    thetas.append(0)
    taus.append(np.zeros(2))
    
    return Snew, thetas,taus
    
    
        