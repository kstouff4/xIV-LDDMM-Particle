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
    
    xc = torch.sum((torch.sum(nu_S,axis=-1)[...,None]*S)/torch.sum(nu_S),axis=0)
    print("xc: ", xc.detach())
    
    Snew = (S - xc)@A.T + tau
    print("min and max Snew")
    print(torch.min(Snew.detach(),axis=0))
    print(torch.max(Snew.detach(),axis=0))
    return Snew

def getSlicesFromWhole(files,zU):
    ## assume files need to be split by z with Z indices taken from original (list of Z's)
    info = np.load(files)
    X = info[info.files[0]]
    nu_X = info[info.files[1]]
    print("min and max")
    print(np.min(X,axis=0))
    print(np.max(X,axis=0))
        
    zs = np.asarray(zU)
    D = (X[...,-1][...,None] - zs[None,...])**2
    slIndex = np.argmin(D,axis=-1)
    
    Slist = []
    nu_Slist = []
    for i in range(len(zU)):
        Slist.append(torch.tensor(X[slIndex == i,0:2]).type(dtype)) # ignore z 
        nu_Slist.append(torch.tensor(nu_X[slIndex == i,...]).type(dtype))
        print("min and max of coordinates and nu_X")
        print(np.min(X[slIndex == i,0:2],axis=0))
        print(np.max(X[slIndex == i,0:2],axis=0))
        print(np.min(nu_X[slIndex == i,...],axis=0))
        print(np.max(nu_X[slIndex == i,...],axis=0))
    return Slist, nu_Slist, slIndex 

def printTransformations(p0,savedir,it):
    thetaTot = []
    tauTotX = []
    tauTotY = []
    tot = 0
    for i in range(0,p0.shape[0],3):
        theta = p0[i].detach().cpu().numpy()
        tau = p0[i+1:i+3].detach().cpu().numpy()
        print("theta is, ", theta)
        print("tau is, ", tau)
        thetaTot.append(theta)
        tauTotX.append(tau[0])
        tauTotY.append(tau[1])
        tot += 1
    thetaTot = np.asarray(thetaTot)*180.0/np.pi
    tauTotX = np.asarray(tauTotX)
    tauTotY = np.asarray(tauTotY)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(thetaTot)),thetaTot,label="angle (deg): mean = {0:.6f}, std = {1:.6f}".format(np.mean(thetaTot),np.std(thetaTot)))
    ax.plot(np.arange(len(tauTotX)),tauTotX,label="X translation (mm): mean = {0:.6f}, std = {1:.6f}".format(np.mean(tauTotX),np.std(tauTotX)))
    ax.plot(np.arange(len(tauTotY)),tauTotY,label="Y translation (mm): mean = {0:.6f}, std = {1:.6f}".format(np.mean(tauTotY),np.std(tauTotY)))
    ax.legend()
    f.savefig(savedir + 'transformations_' + str(it) + '.png',dpi=300)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(thetaTot)),np.abs(thetaTot),label="angle (deg): mean = {0:.6f}, std = {1:.6f}".format(np.mean(np.abs(thetaTot)),np.std(np.abs(thetaTot))))
    ax.plot(np.arange(len(tauTotX)),np.abs(tauTotX),label="X translation (mm): mean = {0:.6f}, std = {1:.6f}".format(np.mean(np.abs(tauTotX)),np.std(np.abs(tauTotX))))
    ax.plot(np.arange(len(tauTotY)),np.abs(tauTotY),label="Y translation (mm): mean = {0:.6f}, std = {1:.6f}".format(np.mean(np.abs(tauTotY)),np.std(np.abs(tauTotY))))
    ax.legend()
    f.savefig(savedir + 'abstransformations_' + str(it) + '.png',dpi=300)

            
    print("sum of angles (deg) and tau")
    print(np.sum(thetaTot))
    print(np.sum(tauTotX))
    print(np.sum(tauTotY))
    print("average of angles and tau")
    print(np.sum(thetaTot)/tot)
    print(np.sum(tauTotX)/tot)
    print(np.sum(tauTotY)/tot)
    
    return

def align(Slist,nu_Slist,sigma,its,savedir,norm=False):
    
    K = GaussLinKernelSingle(sigma,2,nu_Slist[0].shape[-1])
    p0 = torch.zeros((len(Slist)-2)*3).type(dtype).requires_grad_(True)
    print("p0, ", p0.detach())
    lossList = []
    
    if norm:
        c = torch.tensor(10000.0).type(dtype)
    else:
        c = torch.tensor(1.0).type(dtype)
    
    def make_loss(Slist,nu_Slist):
        
        def loss(p0):
            for i in range(len(Slist)-2):
                theta = p0[i*3].view(1,1)
                tau = p0[1+i*3:(i+1)*3].view(1,2)
                if (i == 0):
                    L = (1.0/c)*K(Slist[i],applyRigid2D(Slist[i+1],nu_Slist[i+1],theta,tau),nu_Slist[i],nu_Slist[i+1]).sum()
                else:
                    L += (1.0/c)*K(applyRigid2D(Slist[i],nu_Slist[i],p0[3*(i-1)].view(1,1),p0[1+(i-1)*3:i*3].view(1,2)),applyRigid2D(Slist[i+1],nu_Slist[i+1],theta,tau),nu_Slist[i],nu_Slist[i+1]).sum()
                print("L intermediate, ", L.detach().cpu().numpy())
            L += (1.0/c)*K(applyRigid2D(Slist[-2],nu_Slist[-2],p0[-3].view(1,1),p0[-2:].view(1,2)),Slist[-1],nu_Slist[-2],nu_Slist[-1]).sum()
            return -2.0*L
        return loss
    
    loss = make_loss(Slist,nu_Slist)
    print("p0 ", p0.detach().cpu().numpy().shape)
    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    def closure():
        optimizer.zero_grad()
        Ln = loss(p0)
        print("loss, ", Ln.detach().cpu().numpy())
        lossList.append(np.copy(Ln.detach().cpu().numpy()))
        Ln.backward()
        return Ln
    
    for i in range(its):
        optimizer.step(closure)
        if (np.mod(i,10) == 0):
            printTransformations(p0,savedir,i)
    
    printTransformations(p0,savedir,its)
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossList)),np.asarray(lossList),label="Total Cost, Final = {0:.6f}".format(lossList[-1]))
    ax.legend()
    f.savefig(savedir + 'cost.png',dpi=300)
    
    Snew = [Slist[0]]
    thetas = [0]
    taus = [np.zeros(2)]
    for i in range(len(Slist)-2):
        Snew.append(applyRigid2D(Slist[i+1],nu_Slist[i+1],p0[i*3],p0[1+i*3:(i+1)*3]))
        thetas.append(p0[i*3].detach().cpu().numpy())
        taus.append(p0[1+i*3:(i+1)*3].detach().cpu().numpy())
    Snew.append(Slist[-1])
    thetas.append(0)
    taus.append(np.zeros(2))
    
    return Snew, thetas,taus
    
    
        