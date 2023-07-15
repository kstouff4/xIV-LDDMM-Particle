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

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor #DoubleTensor 
else:
    dtype = torch.FloatTensor

from matplotlib import pyplot as plt
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
    
import sys
from sys import path as sys_path
#sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
#import vtkFunctions as vtf
sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init
from saveState import *
import getOutput as gO

#################################################################################
# Kernels

# (1/2) |u|^2
def GaussKernelHamiltonian(sigma,d,uCoeff):
    qxO,qyO,px,py,wpxO,wpyO = Vi(0,d), Vj(1,d), Vi(2,d), Vj(3,d), Vi(4,1), Vj(5,1)
    #retVal = qxO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        qx,qy,wpx,wpy = qxO/sig, qyO/sig, wpxO/sig, wpyO/sig
        D2 = qx.sqdist(qy)
        K = (-D2 * 0.5).exp()
        h = 0.5*(px*py).sum() + wpy*((qx - qy)*px).sum() - (0.5) * wpx*wpy*(D2 - d) # 1/2 factor included here
        if sInd == 0:
            retVal = (1.0/uCoeff[sInd])*K*h # normalize by N_sigma/(N sigma**2)
        else:
            retVal += (1.0/uCoeff[sInd])*K*h 
    return retVal.sum_reduction(axis=1) #(K*h).sum_reduction(axis=1) #,  h2, h3.sum_reduction(axis=1) 

# |\mu_s - \mu_T |_sigma^2
def GaussLinKernelSingle(sig,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    K = (-D2 / (2.0*sig*sig)).exp() * (u * v).sum()
    return K.sum_reduction(axis=1)

# \sum_sigma \beta/2 * |\mu_s - \mu_T|^2_\sigma
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

# k(x^\sigma - y^\sigma)
def GaussKernelSpaceSingle(sig,d):
    x,y = Vi(0,d)/sig, Vj(1,d)/sig
    D2 = x.sqdist(y)
    K = (-D2 * 0.5).exp()
    return K.sum_reduction(axis=1)

#################################################################
# Compute Controls (A,Tau,U,div(U))
# get A,tau,alpha
def getATauAlpha(px,qx,pw,qw,cA=1.0,cT=1.0,dimEff=3,single=False):
    xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0)) # moving barycenters; should be 1 x 3
    A = ((1.0/(2.0*cA))*(px.T@(qx-xc) - (qx-xc).T@px)).type(dtype) # 3 x N * N x 3
    tau = ((1.0/cT)*(px.sum(dim=0))).type(dtype)
    if (dimEff == 2):
        alpha = ((px*(qx-xc)).sum() + (pw*qw*dimEff).sum()).type(dtype)
        Alpha = torch.eye(3).type(dtype)*alpha
        Alpha[-1,-1] = 0.0 # always scale Z by 0
    elif (dimEff == 3 and single):
        alpha = ((px*(qx-xc)).sum() + (pw*qw*dimEff).sum()).type(dtype)
        Alpha = torch.eye(3).type(dtype)*alpha
    else:
        alpha_xy = 0.5*((px[:,0:2]*(qx-xc)[:,0:2]).sum() + (pw*qw*2.0).sum()).type(dtype)
        alpha_z = ((px[:,-1]*(qx-xc)[:,-1]).sum() + (pw*qw).sum()).type(dtype)
        Alpha = torch.eye(3).type(dtype)*alpha_xy
        Alpha[-1,-1] = alpha_z
    
    return A,tau,Alpha

# compute U and divergence of U
def getU(sigma,d,uCoeff):
    xO,qyO,py,wpyO = Vi(0,d), Vj(1,d), Vj(2,d), Vj(3,1)
    #retVal = xO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x,qy,wpy = xO/sig, qyO/sig, wpyO/sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp() # G x N
        h = py + wpy*(x-qy) # 1 X N x 3
        if sInd == 0:
            retVal = (1.0/uCoeff[sInd])*K*h
        else:
            retVal += (1.0/uCoeff[sInd])*(K*h) #.sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1) # G x 3

def getUdiv(sigma,d,uCoeff):
    xO,qyO,py,wpyO = Vi(0,d), Vj(1,d), Vj(2,d), Vj(3,1)
    #retVal = xO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x,qy,wpy = xO/sig, qyO/sig, wpyO/sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp()
        h = wpy*(d - D2) - ((x-qy)*py).sum()
        if sInd == 0:
            retVal = (1.0/(sig*uCoeff[sInd]))*K*h
        else:
            retVal += (1.0/(sig*uCoeff[sInd]))*(K*h) #.sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1) # G x 1

def defineSupport(Ttilde,eps=0.001):
    # estimate normal vectors and points; assume coronal sections
    zMin = torch.min(Ttilde[:,-1])
    zMax = torch.max(Ttilde[:,-1])
    
    print("zMin: ", zMin)
    print("zMax: ", zMax)
    
    if zMin == zMax:
        zMin = torch.min(Ttilde[:,-2])
        zMax = torch.max(Ttilde[:,-2])
        print("new Zmin: ", zMin)
        print("new Zmax: ", zMax)
        sh = 0
        co = 1
        while (sh < 2):
            lineZMin = Ttilde[torch.squeeze(Ttilde[:,-2] < (zMin + torch.tensor(co*eps).type(dtype))),...]
            lineZMax = Ttilde[torch.squeeze(Ttilde[:,-2] > (zMax - torch.tensor(co*eps).type(dtype))),...]
            sh = min(lineZMin.shape[0],lineZMax.shape[0])
            co += 1
        
        print("lineZMin: ", lineZMin)
        print("lineZMax: ", lineZMax)
        
        tCenter = torch.mean(Ttilde,axis=0)

        a0s = lineZMin[torch.randperm(lineZMin.shape[0])[0:2],...]
        a1s = lineZMax[torch.randperm(lineZMax.shape[0])[0:2],...]
        
        print("a0s: ", a0s)
        print("a1s: ", a1s)
        
        a0 = torch.mean(a0s,axis=0)
        a1 = torch.mean(a1s,axis=0)
        
        print("a0: ", a0)
        print("a1: ", a1)
        
        n0 = torch.tensor([-(a0s[1,1] - a0s[0,1]),(a0s[1,0]-a0s[0,0]),Ttilde[0,-1]]).type(dtype)
        n1 = torch.tensor([-(a1s[1,1] - a1s[0,1]),(a1s[1,0]-a1s[0,0]),Ttilde[0,-1]]).type(dtype)
        if (torch.dot(tCenter - a0,n0) < 0):
            n0 = -1.0*n0

        if (torch.dot(tCenter - a1,n1) < 0):
            n1 = -1.0*n1
 
    else:
        sh = 0
        co = 1
        while (sh < 3):
            sliceZMin = Ttilde[torch.squeeze(Ttilde[:,-1] < (zMin + torch.tensor(co*eps).type(dtype))),...]
            sliceZMax = Ttilde[torch.squeeze(Ttilde[:,-1] > (zMax - torch.tensor(co*eps).type(dtype))),...]
            sh = min(sliceZMin.shape[0],sliceZMax.shape[0])
            co += 1

        print("sliceZMin: ", sliceZMin)
        print("sliceZMax: ", sliceZMax)

        tCenter = torch.mean(Ttilde,axis=0)

        # pick 3 points on each approximate slice and take center and normal vector
        a0s = sliceZMin[torch.randperm(sliceZMin.shape[0])[0:3],...]
        a1s = sliceZMax[torch.randperm(sliceZMax.shape[0])[0:3],...]

        print("a0s: ", a0s)
        print("a1s: ", a1s)

        a0 = torch.mean(a0s,axis=0)
        a1 = torch.mean(a1s,axis=0)

        n0 = torch.cross(a0s[1,...]-a0s[0,...],a0s[2,...] - a0s[0,...])
        if (torch.dot(tCenter - a0,n0) < 0):
            n0 = -1.0*n0

        n1 = torch.cross(a1s[1,...] - a1s[0,...],a1s[2,...] - a1s[0,...])
        if (torch.dot(tCenter - a1,n1) < 0):
            n1 = -1.0*n1
    
    # normalize vectors
    n0 = n0/torch.sqrt(torch.sum(n0**2))
    n1 = n1/torch.sqrt(torch.sum(n1**2))
    
    # ensure dot product with barycenter vector to point is positive, otherwise flip sign of normal
    print("n0: ", n0)
    print("n1: ", n1)
    
    def alphaSupportWeight(qx,lamb):
        return (0.5*torch.tanh(torch.sum((qx-a0)*n0,axis=-1)/lamb) + 0.5*torch.tanh(torch.sum((qx-a1)*n1,axis=-1)/lamb))[...,None]
    
    return alphaSupportWeight
    
###################################################################
def checkEndPoint(lossFunction,p0,p1,q1,d,numS,savedir):
    q = torch.clone(q1).detach().requires_grad_(True).type(dtype)
    p0n = torch.clone(p0).detach().requires_grad_(False).type(dtype)
    p1n = torch.clone(p1).detach().requires_grad_(False).type(dtype)
    
    dLoss = lossFunction(q,p0n[(d+1)*numS:])
    dLoss.backward() # gradient of loss at end point with respect to q1
    print("checking end point condition")
    print(q.grad)
    print(p1[:(d+1)*numS].detach().cpu().numpy())
    print("should equal zero")
    ep = q.grad.detach().cpu().numpy() + p1n[:(d+1)*numS].detach().cpu().numpy()
    print(ep)
    print(np.max(ep))
    print(np.min(ep))
    f,ax = plt.subplots()
    ax.plot(np.arange(numS),ep[:numS],label='x_diff')
    ax.plot(np.arange(numS),ep[numS:2*numS],label='y_diff')
    ax.plot(np.arange(numS),ep[2*numS:3*numS],label='z_diff')
    ax.plot(np.arange(numS),ep[3*numS:],label='w_diff')
    ax.legend()
    f.savefig(savedir+'checkPoint.png',dpi=300)
    
    return

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
def Hamiltonian(K0, sigma, d,numS,cA=1.0,cT=1.0,dimEff=3,single=False):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]

        wpq = pw*qw
        k = K0(qx,qx,px,px,wpq,wpq) # k shape should be N x 1
        # px = N x 3, qx = N x 3, qw = N x 1
        A,tau,Alpha = getATauAlpha(px,qx,pw,qw,cA,cT,dimEff,single) #getAtau( = (1.0/(2*alpha))*(px.T@(qx-qc) - (qx-qc).T@px) # should be d x d
        Anorm = (A*A).sum()  
        Alphanorm = (Alpha*Alpha).sum()
        return k.sum() + (cA/2.0)*Anorm + (cT/2.0)*(tau*tau).sum() + (0.5)*Alphanorm

    return H


def HamiltonianSystem(K0, sigma, d,numS,cA=1.0,cT=1.0,dimEff=3,single=False):
    H = Hamiltonian(K0, sigma, d, numS,cA,cT,dimEff,single)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

# Katie change this to include A and T for the grid 
def HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0,dimEff=3,single=False):
    H = Hamiltonian(K0,sigma,d,numS,cA,cT,dimEff=dimEff,single=single)
    def HS(p,q,qgrid,qgridw,T=None,wT=None):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]
        #pc = p[-d:].view(1,d)
        #qc = q[-d:].view(1,d)
        gx = qgrid.view(-1,d)
        gw = qgridw.view(-1,1)
        Gp,Gq = grad(H(p,q), (p,q), create_graph=True)
        A,tau,Alpha = getATauAlpha(px,qx,pw,qw,dimEff=dimEff,single=single)
        xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0))
        Gg = (getU(sigma,d,uCoeff)(gx,qx,px,pw*qw) + (gx-xc)@A.T + tau + (gx-xc)@Alpha).flatten()
        Ggw = (getUdiv(sigma,d,uCoeff)(gx,qx,px,pw*qw)*gw + Alpha.sum()*gw).flatten()
                              
        if T == None:
            return -Gq,Gp,Gg,Ggw
        else:
            print("including T")
            Tx = T.view(-1,d)
            wTw = wT.view(-1,1)
            Gt = -1.0*(getU(sigma,d,uCoeff)(Tx,qx,px,pw*qw) + (Tx-xc)@A.T + tau + (Tx-xc)@Alpha).flatten()
            Gtw = -1.0*(getUdiv(sigma,d,uCoeff)(Tx,qx,px,pw*qw)*wTw + Alpha.sum()*wTw).flatten()
            return -Gq,Gp,Gg,Ggw,Gt,Gtw
    
    return HS

def HamiltonianSystemBackwards(K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0,dimEff=3,single=False):
    # initial = integration from p0 and q0 just with negative velocity field and parameters (e.g. giving same p's)
    # 06/25 = first try with just putting -px, -pw replacement only --> doesn't work (seems to give incorrect mapping)
    # 06/26 = second try with just changing hamiltonian derivatives in terms of sign (change t --> 1 - t); scale off (too big)
    # 7/01 = third try with starting with -p1, but keeping same integration scheme in terms of relation with hamiltonians
    H = Hamiltonian(K0,sigma,d,numS,cA,cT,dimEff=dimEff,single=single)
    def HS(p,q,T,wT):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]
        Gp,Gq = grad(H(p,q), (p,q), create_graph=True)
        A,tau,Alpha = getATauAlpha(px,qx,pw,qw,dimEff=dimEff,single=single)
        xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0))
        Tx = T.view(-1,d)
        wTw = wT.view(-1,1)
        Gt = (getU(sigma,d,uCoeff)(Tx,qx,px,pw*qw) + (Tx-xc)@A.T + tau + (Tx-xc)@Alpha).flatten()
        Gtw = (getUdiv(sigma,d,uCoeff)(Tx,qx,px,pw*qw)*wTw + Alpha.sum()*wTw).flatten()
        return -Gq,Gp,Gt,Gtw
    
    return HS
        
##################################################################
# Shooting

def Shooting(p0, q0, K0,sigma,d, numS,cA=1.0,cT=1.0,dimEff=3,single=False,nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,sigma,d,numS,cA,cT,dimEff,single), (p0, q0), nt)


def LDDMMloss(K0,sigma, d, numS,gamma,dataloss,piLoss,lambLoss,cA=1.0,cT=1.0,cPi=10.0,dimEff=3,single=False):
    def loss(p0, q0):
        p, q = Shooting(p0[:(d+1)*numS], q0, K0,sigma, d,numS,dimEff=dimEff,single=single)[-1]
        hLoss = (gamma * Hamiltonian(K0, sigma, d,numS,cA,cT,dimEff,single=single)(p0[:(d+1)*numS], q0))
        dLoss = dataloss(q,p0[(d+1)*numS:])
        pLoss = gamma*torch.tensor(cPi).type(dtype)*piLoss(q,p0[(d+1)*numS:-1])
        lLoss = gamma*lambLoss(p0[-1])
        return hLoss, dLoss, pLoss, lLoss
        #return dataloss(q)

    return loss

def ShootingGrid(p0,q0,qGrid,qGridw,K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0,dimEff=3,single=False,nt=10,Integrator=RalstonIntegrator(),T=None,wT=None):
    if T == None:
        return Integrator(HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff,cA,cT,dimEff,single=single), (p0[:(d+1)*numS],q0,qGrid,qGridw),nt)
    else:
        print("T shape adn wT shape")
        print(T.detach().cpu().numpy().shape)
        print(wT.detach().cpu().numpy().shape)
        print("G and wG shape")
        print(qGrid.detach().cpu().numpy().shape)
        print(qGridw.detach().cpu().numpy().shape)
        return Integrator(HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff,cA,cT,dimEff,single=single), (p0[:(d+1)*numS],q0,qGrid,qGridw,T,wT),nt)
    
def ShootingBackwards(p1,q1,T,wT,K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0,dimEff=3,single=False,nt=10,Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystemBackwards(K0,sigma,d,numS,uCoeff,cA,cT,dimEff,single=single), (-p1[:(d+1)*numS],q1,T,wT),nt)

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K,d,numS,supportWeight):
    #print(w_T*zeta_T.cpu().numpy())
    cst = (K(T,T,w_T*zeta_T,w_T*zeta_T)).sum()

    def loss(sS,pi_est):
        # sS will be in the form of q (w_S,S,x_c)
        sSx = sS[numS:].view(-1,d)
        sSw = sS[:numS].view(-1,1)
        wSupport = supportWeight(sSx,pi_est[-1]**2)
        sSw = wSupport*sSw
        pi_ST = (pi_est[:-1].view(zeta_S.shape[-1],zeta_T.shape[-1]))**2
        nu_Spi = (sSw*zeta_S)@pi_ST # Ns x L * L x F                    
            
     
        k1 = K(sSx, sSx, nu_Spi, nu_Spi) 
        k2 = K(sSx, T, nu_Spi, w_T*zeta_T)
              
        return (
            (1.0/2.0)*(cst
            + k1.sum()
            - 2.0 * k2.sum())
        )

    return cst.detach().cpu().numpy(), loss

# pi regularization (KL divergence)
def PiRegularizationSystem(zeta_S,nu_T,numS,d,norm=True):
    nuTconst = torch.sum(nu_T)
    # two alternatives for distribution to compare to 
    if (not norm):
        nu_Tprob = torch.sum(nu_T,dim=0)/torch.sum(nu_T) # compare to overall distribution of features
    else:
        nu_Tprob = (torch.ones((1,nu_T.shape[-1]))/nu_T.shape[-1]).type(dtype) 
    
    def PiReg(q,pi_est):  
        qw = q[:numS].view(-1,1)
        mass_S = torch.sum(qw*zeta_S,dim=0)
        qwSum = torch.sum(qw,dim=0)[None,...]
        pi_ST = (pi_est.view(zeta_S.shape[-1],nu_T.shape[-1]))**2
        pi_S = torch.sum(pi_ST,dim=-1)
        pi_STprob = pi_ST/pi_S[...,None]
        numer = pi_STprob/nu_Tprob
        di = mass_S@(pi_ST*(torch.log(numer)))
        return (1.0/nuTconst)*di.sum()
    return PiReg

# boundary weight regularization 
def supportRestrictionReg(eta0=torch.sqrt(torch.tensor(0.1))):
    def etaReg(eta):
        lamb = (eta/eta0)**2
        reg = lamb*torch.log(lamb) + 1.0 - lamb
        return reg
    return etaReg

##################################################################
# Print out Functions

def printCurrentVariables(p0Curr,itCurr,K0,sigmaRKHS,uCoeff,q0Curr,d,numS,zeta_S,labT,s,m,savedir,supportWeightF,dimEff=3,single=False):
    np.savez(savedir+'p0_iter' + str(itCurr) + '.npz',p0=p0Curr.detach().cpu().numpy())
    pqList = Shooting(p0Curr[:(d+1)*numS], q0Curr, K0,sigmaRKHS, d,numS,dimEff=dimEff,single=single)
    pi_ST = p0Curr[(d+1)*numS:-1].detach().view(zeta_S.shape[-1],labT)
    pi_ST = pi_ST**2
    pi_ST = pi_ST.cpu().numpy()
    print("pi iter " + str(itCurr) + ", ", pi_ST)
    print(np.unique(pi_ST))
    print("non diffeomorphism")
    totA = np.zeros((3,3))
    print("lambda: ", (p0Curr[-1].detach().cpu().numpy())**2)
    for i in range(len(pqList)):
        p = pqList[i][0]
        q = pqList[i][1]
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]
        A,tau,Alpha = getATauAlpha(px,qx,pw,qw,dimEff=dimEff,single=single)
        totA += (0.1)*A.detach().cpu().numpy() # assume 10 time steps?
        print("A, ", A.detach().cpu().numpy())
        print("tau, ", tau.detach().cpu().numpy())
        print("alpha, ", Alpha.detach().cpu().numpy())
    R = sp.linalg.expm(totA)
    print("R, ", R)
    print("det(R), ", np.linalg.det(R))
    
    if (pi_ST.shape[0] > pi_ST.shape[1]*10):
        rat = np.round(pi_ST.shape[0]/pi_ST.shape[1]).astype(int)
        pi_STplot = np.zeros((pi_ST.shape[0],rat*pi_ST.shape[1]))
        for j in range(pi_ST.shape[1]):
            pi_STplot[:,j*rat:(j+1)*rat] = np.squeeze(pi_ST[:,j])[...,None]
    else:
        pi_STplot = pi_ST
    f,ax = plt.subplots()
    im = ax.imshow(pi_STplot)
    f.colorbar(im,ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Label Replicates")
    f.savefig(savedir + 'pi_STiter' + str(itCurr) + '.png',dpi=300)
    
    q = pqList[-1][1]
    D = q[numS:].detach().view(-1,d)
    wS = supportWeightF(D,p0Curr[-1].detach()**2)
    D = D.cpu().numpy()
    muD = q[0:numS].detach().view(-1,1).cpu().numpy()
    nu_D = np.squeeze(muD[0:numS])[...,None]*zeta_S.detach().cpu().numpy()
    zeta_D = nu_D/np.sum(nu_D,axis=-1)[...,None]
    nu_Dpi = nu_D@pi_ST
    Ds = init.resizeData(D,s,m)
    zeta_Dpi = nu_Dpi/np.sum(nu_Dpi,axis=-1)[...,None]
    print("nu_D shape original: ", nu_D.shape)
    
    imageNamesSpi = ['weights', 'maxImageVal']
    imageNamesS = ['weights', 'maxImageVal']
    imageValsSpi = [np.sum(nu_Dpi,axis=-1),np.argmax(nu_Dpi,axis=-1)]
    imageValsS = [np.sum(nu_D,axis=-1),np.argmax(nu_D,axis=-1)]
    for i in range(zeta_Dpi.shape[-1]):
        imageValsSpi.append(zeta_Dpi[:,i])
        imageNamesSpi.append('zeta_' + str(i))
    for i in range(zeta_D.shape[-1]):
        imageValsS.append(zeta_D[:,i])
        imageNamesS.append('zeta_' + str(i))
    gO.writeVTK(Ds,imageValsS,imageNamesS,savedir+'testOutputiter' + str(itCurr) + '_D10.vtk',polyData=None)
    gO.writeVTK(Ds,imageValsSpi,imageNamesSpi,savedir+'testOutputiter' + str(itCurr) + '_Dpi10.vtk',polyData=None)
    

    nu_D = wS.cpu().numpy()*np.squeeze(muD[0:numS])[...,None]*zeta_S.detach().cpu().numpy()
    print("nu_D shape: ", nu_D.shape)
    zeta_D = nu_D/np.sum(nu_D,axis=-1)[...,None]
    nu_Dpi = nu_D@pi_ST
    Ds = init.resizeData(D,s,m)
    zeta_Dpi = nu_Dpi/np.sum(nu_Dpi,axis=-1)[...,None]
    imageValsSpi = [np.sum(nu_Dpi,axis=-1),np.argmax(nu_Dpi,axis=-1),wS]
    imageValsS = [np.sum(nu_D,axis=-1),np.argmax(nu_D,axis=-1),wS]
    imageNamesSpi = ['weights', 'maxImageVal','support_weights']
    imageNamesS = ['weights', 'maxImageVal','support_weights']
    for i in range(zeta_Dpi.shape[-1]):
        imageValsSpi.append(zeta_Dpi[:,i])
        imageNamesSpi.append('zeta_' + str(i))
    for i in range(zeta_D.shape[-1]):
        imageValsS.append(zeta_D[:,i])
        imageNamesS.append('zeta_' + str(i))

    gO.writeVTK(Ds,imageValsS,imageNamesS,savedir+'testOutputiter' + str(itCurr) + '_D10Support.vtk',polyData=None)
    gO.writeVTK(Ds,imageValsSpi,imageNamesSpi,savedir+'testOutputiter' + str(itCurr) + '_Dpi10Support.vtk',polyData=None)

    return pqList[-1][0],pqList[-1][1]
                            

###################################################################
# Optimization

def makePQ(S,nu_S,T,nu_T,Csqpi=torch.tensor(1.0).type(dtype),lambInit=torch.tensor(0.5).type(dtype),Csqlamb=torch.tensor(1.0).type(dtype),norm=True):
    # initialize state vectors based on normalization 
    w_S = nu_S.sum(axis=-1)[...,None].type(dtype)
    w_T = nu_T.sum(axis=-1)[...,None].type(dtype)
    zeta_S = (nu_S/w_S).type(dtype)
    zeta_T = (nu_T/w_T).type(dtype)
    zeta_S[torch.squeeze(w_S == 0),...] = torch.tensor(0.0).type(dtype)
    zeta_T[torch.squeeze(w_T == 0),...] = torch.tensor(0.0).type(dtype)
    numS = w_S.shape[0]
    
    Stilde, Ttilde, s, m = init.rescaleData(S,T)
    
    q0 = torch.cat((w_S.clone().detach().flatten(),Stilde.clone().detach().flatten()),0).requires_grad_(True).type(dtype) # not adding extra element for xc
    
    # two alternatives (linked to variation of KL divergence norm that consider)
    pi_STinit = torch.zeros((zeta_S.shape[-1],zeta_T.shape[-1])).type(dtype)
    nuSTtot = torch.sum(w_T)/torch.sum(w_S)
    if (not norm):
        pi_STinit[:,:] = nu_T.sum(axis=0)/torch.sum(w_S) # feature distribution of target scaled by total mass in source
    else:
        pi_STinit[:,:] = torch.ones((1,nu_T.shape[-1]))/nu_T.shape[-1]
        pi_STinit = pi_STinit*nuSTtot
    print("pi shape ", pi_STinit.shape)
    print("initial Pi ", pi_STinit.detach().cpu().numpy())
    print("unique values in Pi ", np.unique(pi_STinit.detach().cpu().numpy()))
    
    lamb0 = lambInit
    
    p0 = torch.cat((torch.zeros_like(q0),(1.0/Csqpi)*torch.sqrt(pi_STinit).clone().detach().flatten(),(1.0/Csqlamb)*torch.sqrt(lamb0).clone().detach().flatten()),0).requires_grad_(True).type(dtype)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0,numS, Stilde, Ttilde, s, m, pi_STinit,lamb0

def callOptimize(S,nu_S,T,nu_T,sigmaRKHS,sigmaVar,gamma,d,labs, savedir, its=100,kScale = torch.tensor(1.0).type(dtype),cA=1.0,cT=1.0,cS=10.0,cPi=1.0,dimEff=3,Csqpi=torch.tensor(1.0).type(dtype),Csqlamb=torch.tensor(1.0).type(dtype),eta0=torch.sqrt(torch.tensor(0.1)).type(dtype),lambInit=torch.tensor(0.5).type(dtype),single=False,beta=None,loadPrevious=None):
    '''
    Parameters:
        S, nu_S = source image varifold
        T, nu_T = target image varifold
        sigmaRKHS = list of sigmas for RKHS of velocity vector field (assumed Gaussian)
        sigmaVar = list of sigmas for varifold norm (assumed Gaussian)
        gammaA = weight on norm of infinitesimal rotation A (should be based on varifold norm in end decreasing to 0.01)
        gammaT = weight on norm of infinitesimal translation tau (should be based on varifold norm in end decreasing to 0.01)
        gammaU = weight on norm of velocity vector field (should be based on varifold norm in end decreasing to 0.01)
        d = dimensions of space
        labs = dimensions of feature space
        savedir = location to save cost graphs and p0 in
        its = iterations of LBFGS steps (each step has max 10 iterations and 15 evaluations of objective function)
        kScale = multiplicative factor of particles if applying parameter settings for one set of particles to another (larger), with assumption that mass has been conserved
        beta = computed to scale the varifold norm initially to 1; overridden with desired value if not None
        single = True if x,y,z should be same scaling 
    '''
    w_S, w_T,zeta_S,zeta_T,q0,p0,numS,Stilde,Ttilde,s,m, pi_STinit, lamb0 = makePQ(S,nu_S,T,nu_T,Csqpi=Csqpi,lambInit=lambInit,Csqlamb=Csqlamb)
    N = torch.tensor(S.shape[0]).type(dtype)
    s = s.cpu().numpy()
    m = m.cpu().numpy()

    pTilde = torch.zeros_like(p0).type(dtype)
    pTilde[0:numS] = torch.squeeze(torch.tensor(1.0/(torch.sqrt(kScale)*dimEff*w_S))).type(dtype) #torch.sqrt(kScale)*torch.sqrt(kScale)
    pTilde[numS:numS*(d+1)] = torch.tensor(1.0/torch.sqrt(kScale)).type(dtype) #torch.sqrt(kScale)*1.0/(cScale*torch.sqrt(kScale))
    pTilde[(d+1)*numS:-1] = Csqpi
    pTilde[-1] = Csqlamb
    savepref = savedir + 'State_'
    
    supportWeights = defineSupport(Ttilde)
    sW0 = supportWeights(Stilde,lamb0)
    
    if (beta is None):
        # set beta to make ||mu_S - mu_T||^2 = 1
        if len(sigmaVar) == 1:
            Kinit = GaussLinKernelSingle(sig=sigmaVar[0],d=d,l=labs)
            cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
            k1 = Kinit(Stilde, Stilde, (sW0*w_S*zeta_S)@pi_STinit, (sW0*w_S*zeta_S)@pi_STinit)
            k2 = Kinit(Stilde, Ttilde, (sW0*w_S*zeta_S)@pi_STinit, w_T*zeta_T)
            beta = torch.tensor(2.0/(cinit + k1.sum() - 2.0*k2.sum())).type(dtype)
            print("beta is ", beta.detach().cpu().numpy())
            beta = [(0.6/sigmaVar[0])*torch.clone(2.0/(cinit + k1.sum() - 2.0*k2.sum())).type(dtype)] 
        
        # print out indiviual costs
        else:
            print("different varifold norm at beginning")
            beta = []
            for sig in sigmaVar:
                print("sig is ", sig.detach().cpu().numpy())
                Kinit = GaussLinKernelSingle(sig=sig,d=d,l=labs)
                cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
                k1 = Kinit(Stilde, Stilde, (sW0*w_S*zeta_S)@pi_STinit, (sW0*w_S*zeta_S)@pi_STinit).sum()
                k2 = -2.0*Kinit(Stilde, Ttilde, (sW0*w_S*zeta_S)@pi_STinit, w_T*zeta_T).sum()
                beta.append((0.6/sig)*torch.clone(2.0/(cinit + k1 + k2)).type(dtype))
                print("mu source norm ", k1.detach().cpu().numpy())
                print("mu target norm ", cinit.detach().cpu().numpy())
                print("total norm ", (cinit + k1 + k2).detach().cpu().numpy())
    # Compute constants to weigh each kernel norm in RKHS by
    uCoeff = []
    for sig in sigmaRKHS:
        Kinit = GaussKernelSpaceSingle(sig=sig,d=d)
        uCoeff.append(torch.clone((torch.tensor(cS).type(dtype)*Kinit(Stilde,Stilde).sum())/(N*N*sig*sig)).type(dtype))
    for ss in range(len(uCoeff)):
        print("sig is ", sigmaRKHS[ss].detach().cpu().numpy())
        print("uCoeff ", uCoeff[ss].detach().cpu().numpy())

    cst, dataloss = lossVarifoldNorm(Ttilde,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigmaVar,d=d,l=labs,beta=beta),d,numS,supportWeights)
    piLoss = PiRegularizationSystem(zeta_S,nu_T,numS,d)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)
    lambLoss = supportRestrictionReg(eta0)

    loss = LDDMMloss(Kg,sigmaRKHS,d, numS, gamma, dataloss,piLoss,lambLoss,cA,cT,cPi,dimEff,single=single)
    
    saveParams(uCoeff,sigmaRKHS,sigmaVar,beta,d,labs,numS,pTilde,gamma,cA,cT,cPi,dimEff,single,savepref)

    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    # keep track of both losses
    lossListH = []
    lossListDA = []
    lossListPI = []
    lossListL = []
    relLossList = []
    lossOnlyH = []
    lossOnlyDA = []
    def closure():
        optimizer.zero_grad()
        LH,LDA,LPI,LL = loss(p0*pTilde, q0)
        L = LH+LDA+LPI+LL
        '''
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())
        print("loss LPI ", LPI.detach().cpu().numpy())
        '''
        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))
        relLossList.append(np.copy(LDA.detach().cpu().numpy())/cst)
        lossListPI.append(np.copy(LPI.detach().cpu().numpy()))
        lossListL.append(np.copy(LL.detach().cpu().numpy()))
        L.backward()
        return L
    
    def printCost(currIt):    
        # save p0 for every 50th iteration and pi
        f,ax = plt.subplots()
        ax.plot(np.arange(len(lossListH)),np.asarray(lossListH),label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListH[-1]))
        ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Varifold Norm, Final = {0:.6f}".format(lossListDA[-1]))
        ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA)+np.asarray(lossListH),label="Total Cost, Final = {0:.6f}".format(lossListDA[-1]+lossListH[-1]+lossListPI[-1]))
        ax.plot(np.arange(len(lossListH)),np.asarray(lossListPI),label="Pi Cost, Final = {0:.6f}".format(lossListPI[-1]))
        ax.plot(np.arange(len(lossListH)),np.asarray(lossListL),label="Lambda Cost, Final = {0:.6f}".format(lossListL[-1]))
        ax.set_title("Loss")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.legend()
        f.savefig(savedir + 'Cost_' + str(currIt) + '.png',dpi=300)
        return
    
    if (loadPrevious is not None):
        stateVars = loadState(loadPrevious)
        its = stateVars['its'] + its # add the desired iterations on top of remaining previous ones 
        p0 = stateVars['xopt']
        optimizer.load_state_dict(stateVars['optimizer'])
    
    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure) # default of 25 iterations in strong wolfe line search; will compute evals and iters until 25 unless reaches an optimum 
        print("Current Losses", flush=True)
        print("H loss: ", lossListH[-1])
        print("Var loss: ", lossListDA[-1])
        print("Pi lossL ", lossListPI[-1])
        print("Lambda loss ", lossListL[-1])
        
        osd = optimizer.state_dict()
        lossOnlyH.append(np.copy(osd['state'][0]['prev_loss']))
        saveState(osd,its,i,p0,savepref)
        if (i > 0 and np.isnan(lossListH[-1]) or np.isnan(lossListDA[-1])):
            print("Exiting with detected NaN in Loss")
            print("state of optimizer")
            print(osd)
            break
        if (np.mod(i,20) == 0):
            #p0Save = torch.clone(p0).detach()
            optimizer.zero_grad()
            p1,q1 = printCurrentVariables(p0*pTilde,i,Kg,sigmaRKHS,uCoeff,q0,d,numS,zeta_S,labs,s,m,savedir,supportWeights,dimEff=dimEff,single=single)
            printCost(i)
            #checkEndPoint(dataloss,p0*pTilde,p1,q1,d,numS,savedir + 'it' + str(i))
            if (i > 0):
                if (np.allclose(lossListH[-1],lossListH[-2],atol=1e-6,rtol=1e-5) and np.allclose(lossListDA[-1],lossListDA[-2],atol=1e-6,rtol=1e-5)):
                    print("state of optimizer")
                    print(osd)
                    break

    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")
    saveVariables(q0,p0*pTilde,Ttilde,w_T,s,m,savepref)
    
    printCost(its)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyH),label="TotLoss, Final = {0:.6f}".format(lossOnlyH[-1]))
    #ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA),label="Varifold Norm, Final = {0:.2f}".format(lossOnlyDA[-1]))
    #ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA)+np.asarray(lossOnlyH),label="Total Cost, Final = {0:.2f}".format(lossOnlyDA[-1]+lossOnlyH[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'CostOuterIter.png',dpi=300)
    
    
    # Print out deformed states
    #listpq = Shooting(p0, q0, Kg, Kv, sigmaRKHS,d,numS)
    coords = q0[numS:].detach().view(-1,d)
    rangesX = (torch.max(coords[...,0]) - torch.min(coords[...,0]))/100.0
    rangesY = (torch.max(coords[...,1]) - torch.min(coords[...,1]))/100.0
    rangesZ = (torch.max(coords[...,2]) - torch.min(coords[...,2]))/100.0
    
    if (rangesX == 0):
        rangesX = torch.max(coords[...,0])
        xGrid = torch.arange(torch.min(coords[...,0])+1.0)
    else:
        xGrid = torch.arange(torch.min(coords[...,0])-rangesX,torch.max(coords[...,0])+rangesX*2,rangesX)
    if (rangesY == 0):
        rangesY = torch.max(coords[...,1])
        yGrid = torch.arange(torch.min(coords[...,1])+1.0)
    else:
        yGrid = torch.arange(torch.min(coords[...,1])-rangesY,torch.max(coords[...,1])+rangesY*2,rangesY)
    if (rangesZ == 0):
        rangesZ = torch.max(coords[...,2])
        zGrid = torch.arange(torch.min(coords[...,2])+1.0)
    else:
        zGrid = torch.arange(torch.min(coords[...,2])-rangesZ,torch.max(coords[...,2])+rangesZ*2,rangesZ)
    
    XG,YG,ZG = torch.meshgrid((xGrid,yGrid,zGrid),indexing='ij')
    qGrid = torch.stack((XG.flatten(),YG.flatten(),ZG.flatten()),axis=-1).type(dtype)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG)).type(dtype)    

    listpq = ShootingGrid(p0*pTilde,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,uCoeff,cA,cT,dimEff,single=single)
    print("length of pq list is, ", len(listpq))
    Dlist = []
    nu_Dlist = []
    nu_DPilist = []
    Glist = []
    wGlist = []
    Tlist = []
    nuTlist = []
    
    p0T = p0*pTilde
    pi_STfinal = p0T[(d+1)*numS:-1].detach().view(zeta_S.shape[-1],zeta_T.shape[-1]) # shouldn't matter multiplication by pTilde
    pi_STfinal = pi_STfinal**2
    pi_STfinal = pi_STfinal.cpu().numpy()
    print("pi final, ", pi_STfinal)
    print(np.unique(pi_STfinal))
    
    if (pi_STfinal.shape[0] > pi_STfinal.shape[1]*10):
        rat = np.round(pi_STfinal.shape[0]/pi_STfinal.shape[1]).astype(int)
        pi_STplot = np.zeros((pi_STfinal.shape[0],rat*pi_STfinal.shape[1]))
        for j in range(pi_STfinal.shape[1]):
            pi_STplot[:,j*rat:(j+1)*rat] = np.squeeze(pi_STfinal[:,j])[...,None]
    else:
        pi_STplot = pi_STfinal
    f,ax = plt.subplots()
    im = ax.imshow(pi_STplot)
    f.colorbar(im,ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Label Replicates")
    f.savefig(savedir + 'pi_STfinal.png',dpi=300)
    
    for t in range(len(listpq)):
        qnp = listpq[t][1]
        D = qnp[numS:].detach().view(-1,d).cpu().numpy()
        muD = qnp.detach().cpu().numpy()
        nu_D = np.squeeze(muD[0:numS])[...,None]*zeta_S.detach().cpu().numpy()
        nu_Dpi = nu_D@pi_STfinal
        Dlist.append(init.resizeData(D,s,m))
        nu_Dlist.append(nu_D)
        nu_DPilist.append(nu_Dpi)
        
        gt = listpq[t][2]
        G = gt.detach().view(-1,d).cpu().numpy()
        Glist.append(init.resizeData(G,s,m))
        gw = listpq[t][3]
        W = gw.detach().cpu().numpy()
        wGlist.append(W)        
    
    # Shoot Backwards
    # 7/01 = negative of momentum is incorporated into Shooting Backwards function
    listBack = ShootingBackwards(listpq[-1][0],listpq[-1][1],Ttilde.flatten(),w_T.flatten(),Kg,sigmaRKHS,d,numS,uCoeff,dimEff=dimEff,single=single)
    
    for t in range(len(listBack)):
        Tt = listBack[t][2]
        Tlist.append(init.resizeData(Tt.detach().view(-1,d).cpu().numpy(),s,m))
        wTt = listBack[t][3]
        nuTlist.append(wTt.detach().cpu().numpy()[...,None]*zeta_T.detach().cpu().numpy())
    
    # plot p0 as arrows
    #checkEndPoint(dataloss,p0T,listpq[-1][0],listpq[-1][1],d,numS,savedir)
    listSp0 = np.zeros((numS*2,3))
    polyListSp0 = np.zeros((numS,3))
    polyListSp0[:,0] = 2
    polyListSp0[:,1] = np.arange(numS)# +1
    polyListSp0[:,2] = numS + np.arange(numS) #+ 1
    listSp0[0:numS,:] = S.detach().cpu().numpy()
    listSp0[numS:,:] = p0T[numS:(d+1)*numS].detach().view(-1,d).cpu().numpy() + listSp0[0:numS,:]
    featsp0 = np.zeros((numS*2,1))
    featsp0[numS:,:] = p0T[0:numS].detach().view(-1,1).cpu().numpy()
    gO.writeVTK(listSp0,[featsp0],['p0_w'],savedir + 'testOutput_p0.vtk',polyData=polyListSp0)
    pNew = pTilde*p0
    A,tau,Alpha = getATauAlpha(pNew[numS:(d+1)*numS].view(-1,d),q0[numS:].view(-1,d),pNew[:numS].view(-1,1),q0[:numS].view(-1,1),dimEff=dimEff,single=single)
    print("A final, ", A.detach().cpu().numpy())
    print("tau final, ", tau.detach().cpu().numpy())
    print("Alpha final, ", Alpha.detach().cpu().numpy())
    np.savez(savedir + 'testOutput_values.npz',A0=A.detach().cpu().numpy(),tau0=tau.detach().cpu().numpy(),p0=pNew.detach().cpu().numpy(),q0=q0.detach().cpu().numpy(),alpha0=Alpha.detach().cpu().numpy(),pTilde=pTilde.detach().cpu().numpy(),pi_ST=pi_STfinal)
    np.savez(savedir + 'testOutput_targetScaled.npz',T=Ttilde.detach().cpu().numpy(),w_T=w_T.detach().cpu().numpy(),zeta_T=zeta_T.detach().cpu().numpy(),s=s,m=m)
    np.savez(savedir + 'testOutput_Dvars.npz',D=Dlist[-1],nu_D=nu_Dlist[-1],nu_Dpi=nu_DPilist[-1],Td=Tlist[-1],nu_Td=nuTlist[-1])
    
    return Dlist, nu_Dlist, nu_DPilist, Glist, wGlist, Tlist, nuTlist
