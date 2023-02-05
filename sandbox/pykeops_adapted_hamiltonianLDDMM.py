import os
import time
import numpy as np
from numpy import random

import torch
from torch.autograd import grad

import pykeops
import socket
pykeops.set_build_folder("~/.cache/keop" + pykeops.__version__ + "_" + (socket.gethostname()))

from pykeops.torch import Vi, Vj

###################################################################################

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32
################################################################
# Kernels
def GaussKernel(sigma,d):
    x,y = Vi(0,d), Vj(1,d)
    gamma = 1.0 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return K.sum_reduction(axis=1) # N x d

def GaussKernelGrad(sigma,d):
    x,y,b = Vi(0,d), Vj(1,d), Vi(2,1) # one dimensional momentum p_w
    gamma = 1.0 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    Kxy_b = K*b
    g = torch.autograd.grad(Kxy_b, x, create_graph=False)[0]
    print(g.shape)
    return g.sum_reduction(axis=1)

def GaussKernelLaplace(sigma,d):
    x,y,bx,by = Vi(0,d), Vj(1,d), Vi(2,1), Vj(3,1)
    gamma = 1.0/ (sigma*sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    b_Kxy_b = bx*K*by
    g = torch.trace(torch.autograd.grad(torch.autograd.grad(b_Kxy_b,x,create_graph=False)[0],x,create_graph=False))
    return g.sum_reduction(axis=1)

def GaussKernelB(sigma,d):
    # b is px (spatial momentum)
    x, y, b = Vi(0, d), Vj(1, d), Vj(2,d)
    gamma = 1.0 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return (K*b).sum_reduction(axis=1)

def GaussLinKernel(sigma,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    gamma = 1.0 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum()
    return (K).sum_reduction(axis=1)
    
###################################################################
# Integration

def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        '''
        s = []
        x = []
        for y in x0:
            if isinstance(y,tuple):
                s.append(len(y))
                for z in y:
                    x.append(z.clone())
            else:
                s.append(1)
                x.append(y.clone())
        '''
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        ##### KMS ADDED #####
        '''
        nl = []
        c = 0
        for ss in s:
            if ss < 2:
                nl.append(l[c])
                c += 1
            else:
                nnl = []
                for i in range(ss):
                    nnl.append(l[c])
                    c += 1
                nl.append(tuple(nnl))
        '''
        return l

    return f

#################################################################
# Hamiltonian 
def Hamiltonian(K0,K1,K2):
    # K0 = GaussKernelB(x,y,b)
    # K1 = GaussKernelGrad(x,y,bx)
    # K2 = GaussKernelLaplace(x,y,bx,by)
    def H(p, q):
        px = p[:,1:]
        pw = p[:,0]
        qx = q[:,1:]
        qw = q[:,0]
        
        H0 = (px * K0(qx,qx,px)).sum()
        H1 = (px * K1(qx, qx, pw*qw)).sum()
        H2 = (K2(qx,qx,pw*qw, pw*qw)).sum()
        return 0.5 * H0 + H1 - 0.5*H2 # 0.5 * (px * K(qx, qx, px)).sum()

    return H


def HamiltonianSystem(K0,K1,K2):
    H = Hamiltonian(K0,K1,K2)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

##################################################################
# Shooting

def Shooting(p0, q0, K0, K1, K2 nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,K1,K2), (p0, q0), nt)


def Flow(x0, p0, q0, K0, K1, K2, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K0,K1,K2)

    def FlowEq(x, p, q):
        return (K0(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K0, K1, K2 dataloss, gamma=0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K0, K1, K2)[-1]
        return gamma * Hamiltonian(K0, K1, K2)(p0, q0) + dataloss(q)

    return loss

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K):
    cst = (K(T,T,w_T*zeta_T,w_T*zeta_T)).sum()

    def loss(sS):
        # sS will be in the form of q (S,w_S)
        sSx = sS[:,:d]
        sSw = torch.squeeze(sS[:,d:])[...,None]
        return (
            cst
            + (K(sSx, sSx, sSw*zeta_S, sSw*zeta_S)).sum()
            - 2 * (K(sSx, T, sSw*zeta_S, w_T*zeta_T)).sum()
        )

    return loss

###################################################################
# Optimization

def makePQ(S,nu_S,T,nu_T):
    # initialize state vectors based on normalization 
    w_S = nu_S.sum(axis=-1)[...,None]
    w_T = nu_T.sum(axis=-1)[...,None]
    zeta_S = nu_S/w_S
    zeta_T = nu_T/w_T
    q0 = torch.cat((S,w_S),1).requires_grad_(True)
    p0 = torch.zeros_like(q0).requires_grad_(True)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0

def closure():
    optimizer.zero_grad()
    L = loss(p0, q0)
    print("loss", L.detach().cpu().numpy())
    L.backward()
    return L

def callOptimize(S,nu_S,T,nu_T,sigma,d,labs):
    w_S, w_T,zeta_S,zeta_T,q0,p0 = makePQ(S,nu_S,T,nu_T)
    dataloss = lossVarifoldNorm(T,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigma,d=d,l=labs))
    Kv = GaussKernelB(sigma=sigma,d=d)
    gKv = GaussKernelGrad(sigma=sigma,d=d)
    lKv = GaussKernelLaplace(sigma=sigma,d=d)

    loss = LDDMMloss(Kv,gKv,lKv, dataloss)

    optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10)
    print("performing optimization...")
    start = time.time()
    
    for i in range(10):
    print("it ", i, ": ", end="")
    optimizer.step(closure)
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    muD = q0.detach().cpu().numpy()
    nu_D = np.squeeze(muD[:,-1])*zeta_S.detach().cpu().numpy()
    D = muD[:,0:-1]
    return D, nu_D

   