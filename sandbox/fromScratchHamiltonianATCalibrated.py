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

np_dtype = "float64" #"float32"
dtype = torch.cuda.DoubleTensor #FloatTensor 

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
    xi,xj = Vi(0,d)/sig, Vj(1,d)/sig
    D2 = x.sqdist(y)
    K = (-D2 * 0.5).exp()
    return K.sum_reduction(axis=1)

#################################################################
# Compute Controls (A,Tau,U,div(U))
# get A,tau
def getATau(px,qx,qw):
    xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0)) # moving barycenters
    A = ((1.0/(2.0))*(px.T@(qx-xc) - (qx-xc).T@px)).type(dtype) # should be d x d
    print("A is, ", A.detach().cpu().numpy())
    tau = ((px.sum(dim=0))).type(dtype)
    print("tau is, ", tau.detach().cpu().numpy())
    return A,tau

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
        h = d*wpy - (1.0/sig)*((x-qy)*py).sum() - (1.0/sig)*wpy*D2
        if sInd == 0:
            retVal = (1.0/uCoeff[sInd])*K*h
        else:
            retVal += (1.0/uCoeff[sInd])*(K*h) #.sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1) # G x 1


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
def Hamiltonian(K0, sigma, d,numS):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]

        wpq = pw*qw
        k = K0(qx,qx,px,px,wpq,wpq) # k shape should be N x 1

        print("u norm cost is ")
        print(k.detach().cpu().numpy())
        #print("h is, ", h.detach())
        #print("h2 is, ", h2.detach())
        A,tau = getATau(px,qx,qw) #getAtau( = (1.0/(2*alpha))*(px.T@(qx-qc) - (qx-qc).T@px) # should be d x d
        Anorm = (A*A).sum()
        print("Anorm is, ", (torch.clone(gammaA).cpu().numpy()/2.0)*torch.clone(Anorm).detach().cpu().numpy())
        print("tau is, ", torch.clone(tau).detach().cpu().numpy())
        print("tauNorm is, ", (torch.clone(gammaT).cpu().numpy()/2.0)*(np.sum(torch.clone(tau).detach().cpu().numpy()*torch.clone(tau).detach().cpu().numpy())))
        
        return k.sum() + (0.5)*Anorm + (0.5)*(tau*tau).sum()

    return H


def HamiltonianSystem(K0, sigma, d,numS):
    H = Hamiltonian(K0, sigma, d, numS)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

# Katie change this to include A and T for the grid 
def HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff):
    H = Hamiltonian(K0,sigma,d,numS)
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
        A,tau = getATau(px,qx,qw)
        xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0))
        Gg = (getU(sigma,d,uCoeff)(gx,qx,px,pw*qw) + (gx-xc)@A.T + tau).flatten()
        Ggw = (getUdiv(sigma,d,uCoeff)(gx,qx,px,pw*qw)*gw).flatten()
                                                   
        return -Gq,Gp,Gg,Ggw
    
    return HS
    
        
##################################################################
# Shooting

def Shooting(p0, q0, K0,sigma,d, numS,nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,sigma,d,numS), (p0, q0), nt)


def LDDMMloss(K0,sigma, d, numS,gamma,dataloss):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K0,sigma, d,numS)[-1]
        return gamma * Hamiltonian(K0, sigma, d,numS)(p0, q0), dataloss(q)
        #return dataloss(q)

    return loss

def ShootingGrid(p0,q0,qGrid,qGridw,K0,sigma,d,numS,uCoeff,nt=10,Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff), (p0,q0,qGrid,qGridw),nt)

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K,d,numS):
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
            (1.0/2.0)*(cst
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
    
    Stilde, Ttilde, s, m = init.rescaleData(S,T)
    
    q0 = torch.cat((w_S.clone().detach().flatten(),Stilde.clone().detach().flatten()),0).requires_grad_(True).type(dtype) # not adding extra element for xc
    p0 = (torch.zeros_like(q0)).requires_grad_(True).type(dtype)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0,numS, Stilde, Ttilde, s, m

def callOptimize(S,nu_S,T,nu_T,sigmaRKHS,sigmaVar,gamma,d,labs, savedir, its=100,kScale = torch.tensor(1.0).type(dtype),beta=None):
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
    '''
    w_S, w_T,zeta_S,zeta_T,q0,p0,numS,Stilde,Ttilde,s,m = makePQ(S,nu_S,T,nu_T)
    N = torch.tensor(S.shape[0]).type(dtype)
    s = s.cpu().numpy()
    m = m.cpu().numpy()
    print("sigmaRKHS, ", sigmaRKHS)
    print("sigmaVar, ", sigmaVar)
    pTilde = torch.zeros_like(p0).type(dtype)
    pTilde[0:numS] = kScale #torch.sqrt(kScale)*torch.sqrt(kScale)
    pTilde[numS:] = torch.tensor(1.0).type(dtype) #torch.sqrt(kScale)*1.0/(cScale*torch.sqrt(kScale))

    
    if (beta is None):
        # set beta to make ||mu_S - mu_T||^2 = 1
        if len(sigmaVar) == 1:
            Kinit = GaussLinKernelSingle(sig=sigmaVar[0],d=d,l=labs)
            cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
            k1 = Kinit(Stilde, Stilde, w_S*zeta_S, w_S*zeta_S)
            k2 = Kinit(Stilde, Ttilde, w_S*zeta_S, w_T*zeta_T)
            beta = torch.tensor(2.0/(cinit + k1.sum() - 2*k2.sum())).type(dtype)
            print("beta is ", beta.detach().cpu().numpy())
            beta = [torch.clone(2.0/(cinit + k1.sum() - 2*k2.sum())).type(dtype)]
        
        # print out indiviual costs
        else:
            print("different varifold norm at beginning")
            beta = []
            for sig in sigmaVar:
                print("sig is ", sig.detach().cpu().numpy())
                Kinit = GaussLinKernelSingle(sig=sig,d=d,l=labs)
                cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
                k1 = Kinit(Stilde, Stilde, w_S*zeta_S, w_S*zeta_S).sum()
                k2 = -2*Kinit(Stilde, Ttilde, w_S*zeta_S, w_T*zeta_T).sum()
                beta.append(torch.clone(2.0/(cinit + k1 + k2)).type(dtype))
                print("mu source norm ", k1.detach().cpu().numpy())
                print("mu target norm ", cinit.detach().cpu().numpy())
                print("total norm ", (cinit + k1 + k2).detach().cpu().numpy())
    # Compute constants to weigh each kernel norm in RKHS by
    uCoeff = []
    for sig in sigmaRKHS:
        Kinit = GaussKernelSpaceSingle(sigma=sig,d=d)
        uCoeff.append(torch.clone((1.0/(N*N*sig*sig))*Kinit(Stilde,Stilde).sum()).type(dtype))
    for ss in range(len(uCoeff)):
        print("sig is ", sigmaRKHS[ss].detach().cpu().numpy())
        print("uCoeff ", uCoeff[ss].detach().cpu().numpy())

    cst, dataloss = lossVarifoldNorm(Ttilde,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigmaVar,d=d,l=labs,beta=beta),d,numS)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)

    loss = LDDMMloss(Kg,sigmaRKHS,d, numS, gamma, dataloss)

    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=10,tolerance_grad=1e-8,tolerance_change=1e-10)
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
        LH,LDA = loss(p0*pTilde, q0)
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
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListH),label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Varifold Norm, Final = {0:.6f}".format(lossListDA[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA)+np.asarray(lossListH),label="Total Cost, Final = {0:.6f}".format(lossListDA[-1]+lossListH[-1]))
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
    ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyH),label="TotLoss, Final = {0:.6f}".format(lossOnlyH[-1]))
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
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListHdiff),label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListHdiff[-1]))
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListDAdiff),label="Varifold Norm, Final = {0:.6f}".format(lossListDAdiff[-1]))
    ax.plot(np.arange(len(lossListHdiff)),np.asarray(lossListTotdiff),label="Total Cost, Final = {0:.6f}".format(lossListTotdiff[-1]))
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

    listpq = ShootingGrid(p0*pTilde,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,uCoeff)
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
        Dlist.append(init.resizeData(D,s,m))
        nu_Dlist.append(nu_D)
        gt = listpq[t][2]
        G = gt.detach().view(-1,d).cpu().numpy()
        Glist.append(init.resizeData(G,s,m))
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
    pNew = pTilde*p0
    A,tau = getATau(pNew[numS:].view(-1,d),q0[numS:].view(-1,d),q0[:numS].view(-1,1))
    np.savez(savedir + 'testOutput_values.npz',A0=A.detach().cpu().numpy(),tau0=tau.detach().cpu().numpy(),p0=p0.detach().cpu().numpy(),q0=q0.detach().cpu().numpy())    
    
    return Dlist, nu_Dlist, Glist, wGlist
