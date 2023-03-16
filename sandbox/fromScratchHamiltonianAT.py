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

np_dtype = "float32" #"float32"
dtype = torch.cuda.FloatTensor #FloatTensor 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
import initialize as gi


# Kernels
def GaussKernelHamiltonian(sigma,d):
    qxO,qyO,px,py,wpxO,wpyO = Vi(0,d), Vj(1,d), Vi(2,d), Vj(3,d), Vi(4,1), Vj(5,1)
    #retVal = qxO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        qx,qy,wpx,wpy = qxO/sig, qyO/sig, wpxO/sig, wpyO/sig
        D2 = qx.sqdist(qy)
        K = (-D2 * 0.5).exp()
        h = 0.5*(px*py).sum() + wpy*((qx - qy)*px).sum() - (0.5) * wpx*wpy*(D2 - d)
        if sInd == 0:
            retVal = K*h
        else:
            retVal += K*h
    return retVal.sum_reduction(axis=1) #(K*h).sum_reduction(axis=1) #,  h2, h3.sum_reduction(axis=1) 

def GaussKernelHamiltonianExtra(sigma,d,gamma):
    qx,px,py,wpx,qc,pc = Vi(0,d)/sigma,Vi(1,d),Vj(2,d),Vi(3,1)/sigma,Vj(4,d)/sigma,Vj(5,d)
    print("qc, ", qc)
    DC = qx.sqdist(qc)
    K2 = (-DC * 0.5).exp()
    print("K2 ", K2)
    h2 = (0.5*(px*pc).sum() + wpx*((qc-qx)*pc).sum())*K2 #Nx1
    print("h2, ", h2)
    h3 = ((1.0/(2*gamma))*(px*py)).sum() # NxN
    print("h3, ", h3)
    return h3.sum_reduction() + h2.sum()

def GaussKernelU(sigma,d):
    xO,qyO,py,wpyO = Vi(0,d), Vj(1,d), Vj(2,d), Vj(3,1)
    #retVal = xO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x,qy,wpy = xO/sig, qyO/sig, wpyO/sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp() # G x N
        h = py + wpy*(x-qy) # 1 X N x 3
        if sInd == 0:
            retVal = K*h
        else:
            retVal += (K*h) #.sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1) # G x 3

def GaussKernelUdiv(sigma,d):
    xO,qyO,py,wpyO = Vi(0,d), Vj(1,d), Vj(2,d), Vj(3,1)
    #retVal = xO.sqdist(qyO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x,qy,wpy = xO/sig, qyO/sig, wpyO/sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp()
        h = d*wpy - (1.0/sig)*((x-qy)*py).sum() - (1.0/sig)*wpy*D2
        if sInd == 0:
            retVal = K*h
        else:
            retVal += (K*h) #.sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1) # G x 1

def GaussLinKernel(sigma,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    K = (-D2 / (2.0*sigma*sigma)).exp() * (u * v).sum() 
    return (K).sum_reduction(axis=1)

def GaussKernelB(sigma,d):
    # b is px (spatial momentum)
    xO, yO, b = Vi(0, d), Vj(1, d), Vj(2,d)
    #retVal = xO.sqdist(yO)*torch.tensor(0).type(dtype)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x,y = xO/sig, yO/sig
        D2 = x.sqdist(y)
        K = (-D2 * 0.5).exp()
        if sInd == 0:
            retVal = K*b
        else:
            retVal += K*b
    return retVal.sum_reduction(axis=1)


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
def Hamiltonian(K0, sigma, d,numS,gammaA,gammaT,gammaU):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]
        #pc = p[-d:].view(1,d) # 1 x d
        #qc = q[-d:].view(1,d) # 1 x d
        
        #h = 0.5*Vi(px)*Vj(px).sum() - (1.0/sigma) * Vj(pw)*Vj(qw)*((Vi(qx) - Vj(qx))*Vi(px)).sum() - (1.0/(2*sigma**2)) * Vi(pw*qw)*Vj(pw*qw)*(Vi(px).sqdist(Vj(px)) - d)
        #print(h)
        '''
        H0 = (px * K0(qx,qx,px)).sum()
        H1 = (px * K1(qx, qx, pw*qw)).sum()
        H2 = (K2(qx,qx,pw*qw, pw*qw)).sum()
        return 0.5 * H0 + H1 - 0.5*H2 # 0.5 * (px * K(qx, qx, px)).sum()
        '''
        #return ((h*K0(qx,qx)).sum_reduction(axis=1)).sum()
        wpq = pw*qw
        k = K0(qx,qx,px,px,wpq,wpq) # k shape should be N x 1
        '''
        qxI,pxI,pyJ,wpxI,qcJ,pcJ = Vi(qx)/sigma,Vi(px),Vj(px),Vi(wpq)/sigma,Vj(qc)/sigma,Vj(pc)
        DC = qxI.sqdist(qcJ)
        K2 = (-DC * 0.5).exp()
        h2 = (0.5*(pxI*pcJ).sum() + wpxI*((qcJ-qxI)*pcJ).sum())*K2 #Nx1
        h3 = ((1.0/(2*gamma))*(pxI*pyJ)).sum() # NxN
        h = h3.sum(dim=1)
        hh = h2.sum(dim=0)
        h = h.sum(dim=0) + hh
        '''
        #h = GaussKernelHamiltonianExtra(sigma=sigma,d=d,gamma=gamma)(qx,px,px,wpq,qc,pc)
        print("k is ")
        print(k.detach().cpu().numpy())
        #print("h is, ", h.detach())
        #print("h2 is, ", h2.detach())
        A,tau = getATau(px,qx,qw,gammaA,gammaT) #getAtau( = (1.0/(2*alpha))*(px.T@(qx-qc) - (qx-qc).T@px) # should be d x d
        Anorm = (A*A).sum()
        print("Anorm is, ", (torch.clone(gammaA).cpu().numpy()/2.0)*torch.clone(Anorm).detach().cpu().numpy())
        print("tauNorm is, ", (torch.clone(gammaT).cpu().numpy()/2.0)*(np.sum(torch.clone(tau).detach().cpu().numpy()*torch.clone(tau).detach().cpu().numpy())))

        #print("Anorm, ", Anorm)
        #h2 = (px*((qx-qc)@A.T)).sum()
        #print("h2, ", h2)
        
        return (gammaU)*k.sum() + (gammaA/2.0)*Anorm + (gamma/2.0)*(tau*tau).sum() #h.sum() + 0.5*torch.sum(pc*pc) + torch.sum(h2) 

    return H

def getATau(px,qx,qw,alpha,gamma):
    xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0)) # moving barycenters
    A = ((1.0/(2.0*alpha))*(px.T@(qx-xc) - (qx-xc).T@px)).type(dtype) # should be d x d
    print("A is, ", A.detach().cpu().numpy())
    tau = ((1.0/gamma)*(px.sum(dim=0))).type(dtype)
    print("tau is, ", tau.detach().cpu().numpy())
    return A,tau

def HamiltonianSystem(K0, sigma, d,numS,gammaA,gammaT,gammaU):
    H = Hamiltonian(K0, sigma, d, numS,gammaA,gammaT,gammaU)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

# Katie change this to include A and T for the grid 
def HamiltonianSystemGrid(K0,sigma,d,numS,gammaA,gammaT,gammaU):
    H = Hamiltonian(K0,sigma,d,numS,gammaA,gammaT,gammaU)
    def HS(p,q,qgrid,qgridw):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]
        #pc = p[-d:].view(1,d)
        #qc = q[-d:].view(1,d)
        gx = qgrid.view(-1,d)
        gw = qgridw.view(-1,1)
        Gp,Gq = grad(H(p,q), (p,q), create_graph=True)
        A,tau = getATau(px,qx,qw,gammaA,gammaT)
        xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0))
        '''                                              
        gxt = Vi(gx)/sigma
        qct = Vj(qc)/sigma
        pct = Vj(pc)
        K2 = (-gxt.sqdist(qct)*0.5).exp()
        print("K2, ", K2)

        Dc = (((gxt-qct)/sigma)*pct).sum()
        print("Dc, ", Dc)
        Gg = (GaussKernelU(sigma,d)(gx,qx,px,pw*qw) + (K2.sum()*pct).sum(dim=1) + gx@A.T + tau).flatten() # + A(qgrid-x_c) + tau
        Ggw = ((GaussKernelUdiv(sigma,d)(gx,qx,px,pw*qw) - (K2.sum()*Dc).sum(dim=1))*gw).flatten()
        '''
        Gg = (GaussKernelU(sigma,d)(gx,qx,px,pw*qw) + (gx-xc)@A.T + tau).flatten()
        Ggw = (GaussKernelUdiv(sigma,d)(gx,qx,px,pw*qw)*gw).flatten()
                                                   
        return -Gq,Gp,Gg,Ggw
    
    return HS
    
        
##################################################################
# Shooting

def Shooting(p0, q0, K0, K1, sigma,d, numS,gammaA,gammaT,gammaU,nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,sigma,d,numS,gammaA,gammaT,gammaU), (p0, q0), nt)


def LDDMMloss(K0, K1, sigma, d, numS,gammaA,gammaT,gammaU,dataloss, c=1.0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K0, K1, sigma, d,numS,gammaA,gammaT,gammaU)[-1]
        return c * Hamiltonian(K0, sigma, d,numS,gammaA,gammaT,gammaU)(p0, q0), dataloss(q)
        #return dataloss(q)

    return loss

def ShootingGrid(p0,q0,qGrid,qGridw,K0,sigma,d,numS,gammaA,gammaT,gammaU,nt=10,Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystemGrid(K0,sigma,d,numS,gammaA,gammaT,gammaU), (p0,q0,qGrid,qGridw),nt)

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K,d,numS,beta):
    #print(w_T*zeta_T.cpu().numpy())
    cst = (K(T,T,w_T*zeta_T,w_T*zeta_T)).sum()
    print("cst is ")
    print(cst.detach().cpu().numpy())

    def loss(sS):
        # sS will be in the form of q (w_S,S,x_c)
        sSx = sS[numS:].view(-1,d)
        sSw = sS[:numS].view(-1,1)
     
        k1 = K(sSx, sSx, sSw*zeta_S, sSw*zeta_S) 
        print("ks")
        print(k1.detach().cpu().numpy())
        k2 = K(sSx, T, sSw*zeta_S, w_T*zeta_T)
        print(k2.detach().cpu().numpy())
              
        return (
            (beta/2.0)*(cst
            + k1.sum()
            - 2 * k2.sum())
        )

    return cst.detach().cpu().numpy(), loss

###################################################################
# Optimization

def makePQ(S,nu_S,T,nu_T):
    # initialize state vectors based on normalization 
    w_S = nu_S.sum(axis=-1)[...,None].type(dtype)
    w_T = nu_T.sum(axis=-1)[...,None].type(dtype)
    zeta_S = (nu_S/w_S).type(dtype)
    zeta_T = (nu_T/w_T).type(dtype)
    numS = w_S.shape[0]
    
    Stilde, Ttilde, s, m = gi.rescaleData(S,T)
    
    q0 = torch.cat((w_S.clone().detach().flatten(),Stilde.clone().detach().flatten()),0).requires_grad_(True).type(dtype) # not adding extra element for xc
    p0 = (torch.zeros_like(q0)).requires_grad_(True).type(dtype)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0,numS, Stilde, Ttilde, s, m

def callOptimize(S,nu_S,T,nu_T,sigmaRKHS,sigmaVar,gammaA,gammaT,gammaU,d,labs, savedir, its=100,beta=None):
    w_S, w_T,zeta_S,zeta_T,q0,p0,numS,Stilde,Ttilde,s,m = makePQ(S,nu_S,T,nu_T)
    N = torch.tensor(S.shape[0]).type(dtype)
    print("sigmaRKHS, ", sigmaRKHS)
    print("sigmaVar, ", sigmaVar)
    
    if (beta is None):
        # set beta to make ||mu_S - mu_T||^2 = 1
        Kinit = GaussLinKernel(sigma=sigmaVar,d=d,l=labs)
        cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
        k1 = Kinit(Stilde, Stilde, w_S*zeta_S, w_S*zeta_S)
        k2 = Kinit(Stilde, Ttilde, w_S*zeta_S, w_T*zeta_T)
        beta = 2.0/(cinit + k1.sum() - 2*k2.sum())
    
    cst, dataloss = lossVarifoldNorm(Ttilde,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigmaVar,d=d,l=labs),d,numS,beta=beta)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d)
    Kv = GaussKernelB(sigma=sigmaRKHS,d=d)

    loss = LDDMMloss(Kg,Kv,sigmaRKHS,d, numS, gammaA,gammaT,gammaU, dataloss)

    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=10)
    print("performing optimization...")
    start = time.time()
    
    # keep track of both losses
    lossListH = []
    lossListDA = []
    relLossList = []
    lossOnlyH = []
    lossOnlyDA = []
    def closure():
        optimizer.zero_grad()
        LH,LDA = loss(p0/torch.sqrt(N), q0)
        L = LH+LDA
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())
        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))
        relLossList.append(np.copy(LDA.detach().cpu().numpy())/cst)
        L.backward()
        return L
    
    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure) # default of 25 iterations in strong wolfe line search; will compute evals and iters until 25 unless reaches an optimum 
        print("state of optimizer")
        osd = optimizer.state_dict()
        print(osd)
        '''
        with torch.no_grad():
            LH,LDA = loss(p0,q0)
        lossOnlyH.append(np.copy(LH.detach().cpu().numpy()))
        lossOnlyDA.append(np.copy(LDA.detach().cpu().numpy()))
        '''
        lossOnlyH.append(np.copy(osd['state'][0]['prev_loss']))
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListH),label="H($q_0$,$p_0$), Final = {0:.2f}".format(lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Varifold Norm, Final = {0:.2f}".format(lossListDA[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA)+np.asarray(lossListH),label="Total Cost, Final = {0:.2f}".format(lossListDA[-1]+lossListH[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'Cost.png',dpi=300)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA/cst),label="Varifold Norm, Final = {0:.2f}".format(lossListDA[-1]/cst))
    ax.set_title("Relative Loss Varifold Norm")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'RelativeLossVarifoldNorm.png',dpi=300)
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyH),label="TotLoss, Final = {0:.2f}".format(lossOnlyH[-1]))
    #ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA),label="Varifold Norm, Final = {0:.2f}".format(lossOnlyDA[-1]))
    #ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA)+np.asarray(lossOnlyH),label="Total Cost, Final = {0:.2f}".format(lossOnlyDA[-1]+lossOnlyH[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'CostOuterIter.png',dpi=300)
    
    lossListHdiff = []
    lossListDAdiff = []
    lossListTotdiff = []
    for i in range(len(lossListH)-1):
        lossListHdiff.append(lossListH[i+1]-lossListH[i])
        lossListDAdiff.append(lossListDA[i+1]-lossListDA[i])
        lossListTotdiff.append(lossListHdiff[i]+lossListDAdiff[i])
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListHdiff),label="H($q_0$,$p_0$), Final = {0:.2f}".format(lossListHdiff[-1]))
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListDAdiff),label="Varifold Norm, Final = {0:.2f}".format(lossListDAdiff[-1]))
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListTotdiff),label="Total Cost, Final = {0:.2f}".format(lossListTotdiff[-1]))
    ax.set_title("Difference in Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'CostDifferences.png',dpi=300)

    print("loss differencesH")
    print(lossListHdiff)
    print("loss differences DA")
    print(lossListDAdiff)
    print("loss differences Tot")
    print(lossListTotdiff)
    
    
    # Print out deformed states
    #listpq = Shooting(p0, q0, Kg, Kv, sigmaRKHS,d,numS)
    coords = q0[numS:].detach().view(-1,d)
    rangesX = (torch.max(coords[...,0]) - torch.min(coords[...,0]))/100.0
    rangesY = (torch.max(coords[...,1]) - torch.min(coords[...,1]))/100.0
    rangesZ = (torch.max(coords[...,2]) - torch.min(coords[...,2]))/100.0
                               
    xGrid = torch.arange(torch.min(coords[...,0])-rangesX,torch.max(coords[...,0])+rangesX*2,rangesX)
    yGrid = torch.arange(torch.min(coords[...,1])-rangesY,torch.max(coords[...,1])+rangesY*2,rangesY)
    zGrid = torch.arange(torch.min(coords[...,2])-rangesZ,torch.max(coords[...,2])+rangesZ*2,rangesZ)
    XG,YG,ZG = torch.meshgrid((xGrid,yGrid,zGrid),indexing='ij')
    qGrid = torch.stack((XG.flatten(),YG.flatten(),ZG.flatten()),axis=-1).type(dtype)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG)).type(dtype)
    listpq = ShootingGrid(p0,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,gammaA,gammaT,gammaU)
    print("length of pq list is, ", len(listpq))
    Dlist = []
    nu_Dlist = []
    Glist = []
    wGlist = []
    for t in range(len(listpq)):
        qnp = listpq[t][1]
        D = qnp[numS:].detach().view(-1,d).cpu().numpy()
        muD = qnp.detach().cpu().numpy()
        nu_D = np.squeeze(muD[0:numS])[...,None]*zeta_S.detach().cpu().numpy()
        Dlist.append(gi.resizeData(D,s,m))
        nu_Dlist.append(nu_D)
        gt = listpq[t][2]
        G = gt.detach().view(-1,d).cpu().numpy()
        Glist.append(gi.resizeData(G,s,m))
        gw = listpq[t][3]
        W = gw.detach().cpu().numpy()
        wGlist.append(W)
        # plot p0 as arrows
    listSp0 = np.zeros((numS*2,3))
    polyListSp0 = np.zeros((numS,3))
    polyListSp0[:,0] = 2
    polyListSp0[:,1] = np.arange(numS)# +1
    polyListSp0[:,2] = numS + np.arange(numS) #+ 1
    listSp0[0:numS,:] = S.detach().cpu().numpy()
    listSp0[numS:,:] = p0[numS:].detach().view(-1,d).cpu().numpy() + listSp0[0:numS,:]
    featsp0 = np.zeros((numS*2,1))
    featsp0[numS:,:] = p0[0:numS].detach().view(-1,1).cpu().numpy()
    vtf.writeVTK(listSp0,[featsp0],['p0_w'],savedir + 'testOutput_p0.vtk',polyData=polyListSp0)
    A,tau = getATau(p0[numS:].view(-1,d),q0[numS:].view(-1,d),q0[:numS].view(-1,1),gammaA,gammaT)
    np.savez(savedir + 'testOutput_values.npz',A0=A.detach().cpu().numpy(),tau0=tau.detach().cpu().numpy(),p0=p0.detach().cpu().numpy(),q0=q0.detach().cpu().numpy())    
    
    return Dlist, nu_Dlist, Glist, wGlist
