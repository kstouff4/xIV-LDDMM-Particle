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
def GaussKernelHamiltonian(sigma,d):
    qx,qy,px,py,wpx,wpy = Vi(0,d)/sigma, Vj(1,d)/sigma, Vi(2,d), Vj(3,d),Vi(4,1)/sigma,Vj(5,1)/sigma
    h = 0.5*(px*py).sum() - wpy*((qx - qy)*px).sum() - (0.5) * wpx*wpy*(qx.sqdist(qy) - d)
    #h = 0.5*px*py.sum(dim=2) - (1.0/sigma) * wpy*((qx - qy)*px).sum(dim=2) - (1.0/(2*sigma**2)) * wpx*wpy*(px.sqdist(py) - d).sum(dim=2)
    gamma = 0.5
    D2 = qx.sqdist(qy)
    print("h")
    print(h)
    print("D2")
    print(D2)
    K = (-D2 * gamma).exp()
    print("h and K shape")
    print(h.shape)
    print(K.shape)
    return (K*h).sum_reduction(axis=1) # N x M 

def GaussKernelGrad(sigma,d):
    # not finished 
    x,y,b = Vi(0,d)/sigma, Vj(1,d)/sigma, Vi(2,1) # \tilde{x} = x / sigma one dimensional momentum p_w
    gamma = 0.5
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    Kxy_b = K*b
    print(Kxy_b.shape)
    print(x.shape)
    print(b.shape)
    g = Kxy_b.grad(x,1.)
    #(g,) = torch.autograd.grad([Kxy_b], [x], 1., create_graph=False)
    print(g.shape)
    return g.sum_reduction(axis=1)

def GaussKernelLaplace(sigma,d):
    x,y,bx,by = Vi(0,d), Vj(1,d), Vi(2,1), Vj(3,1)
    gamma = 1.0/ (sigma*sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    b_Kxy_b = bx*K*by
    g = torch.trace(b_Kxy_b.grad(x,1.).grad(x,1.))
    #g = torch.trace(torch.autograd.grad(torch.autograd.grad(b_Kxy_b,[x],1.,create_graph=True),[x],1.,create_graph=False))
    return g.sum_reduction(axis=1)

def GaussKernelB(sigma,d):
    # b is px (spatial momentum)
    x, y, b = Vi(0, d)/sigma, Vj(1, d)/sigma, Vj(2,d)
    gamma = 0.5
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return (K*b).sum_reduction(axis=1)

def GaussLinKernel(sigma,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    gamma = 1.0 / (2.0*sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum()
    return (K).sum_reduction(axis=1)
    
###################################################################
# Integration

def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
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
        return l

    return f

#################################################################
# Hamiltonian 
def Hamiltonian(K0, sigma, d):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[:,1:]
        pw = torch.squeeze(p[:,0])[...,None]
        qx = q[:,1:]
        qw = torch.squeeze(q[:,0])[...,None]
        
        #h = 0.5*Vi(px)*Vj(px).sum() - (1.0/sigma) * Vj(pw)*Vj(qw)*((Vi(qx) - Vj(qx))*Vi(px)).sum() - (1.0/(2*sigma**2)) * Vi(pw*qw)*Vj(pw*qw)*(Vi(px).sqdist(Vj(px)) - d)
        #print(h)
        '''
        H0 = (px * K0(qx,qx,px)).sum()
        H1 = (px * K1(qx, qx, pw*qw)).sum()
        H2 = (K2(qx,qx,pw*qw, pw*qw)).sum()
        return 0.5 * H0 + H1 - 0.5*H2 # 0.5 * (px * K(qx, qx, px)).sum()
        '''
        #return ((h*K0(qx,qx)).sum_reduction(axis=1)).sum()
        k = (K0(qx,qx,px,px,pw*qw,pw*qw)).sum()
        print("k is ")
        print(k.detach().cpu().numpy())
        return k

    return H


def HamiltonianSystem(K0, sigma, d):
    H = Hamiltonian(K0, sigma, d)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        print("Gp and Gq are ")
        print(Gp.detach().cpu().numpy())
        print(Gq.detach().cpu().numpy())
        return -Gq, Gp

    return HS

##################################################################
# Shooting

def Shooting(p0, q0, K0, K1, sigma,d, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,sigma,d), (p0, q0), nt)


def Flow(x0, p0, q0, K0, K1, sigma, d, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K0,sigma,d)

    def FlowEq(x, p, q):
        return (K1(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K0, K1, sigma, d, dataloss, gamma=0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K0, K1, sigma, d)[-1]
        print("p,q after shooting ") 
        print(p.detach().cpu().numpy())
        print(q.detach().cpu().numpy())
        return gamma * Hamiltonian(K0, sigma, d)(p0, q0) + dataloss(q)

    return loss

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K,beta):
    cst = (K(T,T,w_T*zeta_T,w_T*zeta_T)).sum()
    print("cst is ")
    print(cst.detach().cpu().numpy())

    def loss(sS):
        # sS will be in the form of q (w_S,S)
        sSx = sS[:,1:]
        sSw = torch.squeeze(sS[:,0])[...,None]
        k1 = (K(sSx, sSx, sSw*zeta_S, sSw*zeta_S)).sum() 
        print("ks")
        print(k1.detach().cpu().numpy())
        k2 = (K(sSx, T, sSw*zeta_S, w_T*zeta_T)).sum()
        print(k2.detach().cpu().numpy())
              
        return (
            (beta/2.0)*(cst
            + k1
            - 2 * k2)
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
    q0 = torch.cat((w_S.clone().detach(),S.clone().detach()),1).requires_grad_(True)
    print("q0 shape")
    print(q0.shape)
    p0 = (torch.zeros_like(q0)+0.5).requires_grad_(True)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0

def callOptimize(S,nu_S,T,nu_T,sigma,d,labs):
    w_S, w_T,zeta_S,zeta_T,q0,p0 = makePQ(S,nu_S,T,nu_T)
    dataloss = lossVarifoldNorm(T,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigma,d=d,l=labs),beta=0.1)
    Kg = GaussKernelHamiltonian(sigma=sigma,d=d)
    Kv = GaussKernelB(sigma=sigma,d=d)

    loss = LDDMMloss(Kg,Kv,sigma,d, dataloss)

    optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10)
    print("performing optimization...")
    start = time.time()
    
    def closure():
        optimizer.zero_grad()
        L = loss(p0, q0)
        print("loss", L.detach().cpu().numpy())
        L.backward()
        return L
    
    for i in range(10):
        print("it ", i, ": ", end="")
        optimizer.step(closure)
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    muD = q0.detach().cpu().numpy()
    nu_D = np.squeeze(muD[:,0])[...,None]*zeta_S.detach().cpu().numpy()
    D = muD[:,1:]
    return D, nu_D

   