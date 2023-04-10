import os
import time
import numpy as np
from numpy import random

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
# get A,tau
def getATau(px,qx,qw,cA=1.0,cT=1.0):
    xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0)) # moving barycenters; should be 1 x 3
    A = ((1.0/(2.0*cA))*(px.T@(qx-xc) - (qx-xc).T@px)).type(dtype) # 3 x N * N x 3
    tau = ((1.0/cT)*(px.sum(dim=0))).type(dtype)
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
        h = wpy*(d - D2) - ((x-qy)*py).sum()
        if sInd == 0:
            retVal = (1.0/(sig*uCoeff[sInd]))*K*h
        else:
            retVal += (1.0/(sig*uCoeff[sInd]))*(K*h) #.sum_reduction(axis=1)
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
def Hamiltonian(K0, sigma, d,numS,cA=1.0,cT=1.0):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[numS:].view(-1,d)
        pw = p[:numS].view(-1,1)
        qx = q[numS:].view(-1,d)
        qw = q[:numS].view(-1,1) #torch.squeeze(q[:numS])[...,None]

        wpq = pw*qw
        k = K0(qx,qx,px,px,wpq,wpq) # k shape should be N x 1

        print("u norm cost is ")
        print(k.detach().sum().cpu().numpy())
        # px = N x 3, qx = N x 3, qw = N x 1
        A,tau = getATau(px,qx,qw,cA,cT) #getAtau( = (1.0/(2*alpha))*(px.T@(qx-qc) - (qx-qc).T@px) # should be d x d
        Anorm = (A*A).sum()
        print("tau is, ", torch.clone(tau).detach().cpu().numpy())
        print("tauNormSquared is, ", (np.sum(torch.clone(tau).detach().cpu().numpy()*torch.clone(tau).detach().cpu().numpy())))
        
        #xc = (qw*qx).sum(dim=0)/(qw.sum(dim=0)) # moving barycenters; should be 1 x 3
        #AnormN = (px*((qx-xc)@A.T)).sum()
        print("A is ", torch.clone(A).detach().cpu().numpy())
        print("AnormSquared ", Anorm.detach().cpu().numpy())
        
        return k.sum() + (cA/2.0)*Anorm + (cT/2.0)*(tau*tau).sum()

    return H


def HamiltonianSystem(K0, sigma, d,numS,cA=1.0,cT=1.0):
    H = Hamiltonian(K0, sigma, d, numS,cA,cT)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

# Katie change this to include A and T for the grid 
def HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0):
    H = Hamiltonian(K0,sigma,d,numS,cA,cT)
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

def Shooting(p0, q0, K0,sigma,d, numS,cA=1.0,cT=1.0,nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0,sigma,d,numS,cA,cT), (p0, q0), nt)


def LDDMMloss(K0,sigma, d, numS,gamma,dataloss,piLoss,cA=1.0,cT=1.0,cPi=10.0):
    def loss(p0, q0):
        p, q = Shooting(p0[:(d+1)*numS], q0, K0,sigma, d,numS)[-1]
        return (gamma * Hamiltonian(K0, sigma, d,numS,cA,cT)(p0[:(d+1)*numS], q0)), dataloss(q,p0[(d+1)*numS:]), gamma*torch.tensor(cPi).type(dtype)*piLoss(q,p0[(d+1)*numS:])
        #return dataloss(q)

    return loss

def ShootingGrid(p0,q0,qGrid,qGridw,K0,sigma,d,numS,uCoeff,cA=1.0,cT=1.0,nt=10,Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystemGrid(K0,sigma,d,numS,uCoeff,cA,cT), (p0[:(d+1)*numS],q0,qGrid,qGridw),nt)

#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T,w_T,zeta_T,zeta_S,K,d,numS):
    #print(w_T*zeta_T.cpu().numpy())
    cst = (K(T,T,w_T*zeta_T,w_T*zeta_T)).sum()

    def loss(sS,pi_est):
        # sS will be in the form of q (w_S,S,x_c)
        sSx = sS[numS:].view(-1,d)
        sSw = sS[:numS].view(-1,1)
        pi_ST = (pi_est.view(zeta_S.shape[-1],zeta_T.shape[-1]))**2
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
        nu_Tprob = torch.ones((1,nu_T.shape[-1]))/nu_T.shape[-1] 
    
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

##################################################################
# Print out Functions

def printCurrentVariables(p0Curr,itCurr,K0,sigmaRKHS,uCoeff,q0Curr,d,numS,zeta_S,labT,s,m,savedir):
    np.savez(savedir+'p0_iter' + str(itCurr) + '.npz',p0=p0Curr.detach().cpu().numpy())
    p,q = Shooting(p0Curr[:(d+1)*numS], q0Curr, K0,sigmaRKHS, d,numS)[-1]
    pi_ST = p0Curr[(d+1)*numS:].detach().view(zeta_S.shape[-1],labT)
    pi_ST = pi_ST**2
    pi_ST = pi_ST.cpu().numpy()
    print("pi final, ", pi_ST)
    f,ax = plt.subplots()
    im = ax.imshow(pi_ST)
    f.colorbar(im,ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Labels")
    f.savefig(savedir + 'pi_STiter' + str(itCurr) + '.png',dpi=300)
    
    D = q[numS:].detach().view(-1,d).cpu().numpy()
    muD = q[0:numS].detach().view(-1,1).cpu().numpy()
    nu_D = np.squeeze(muD[0:numS])[...,None]*zeta_S.detach().cpu().numpy()
    zeta_D = nu_D/np.sum(nu_D,axis=-1)[...,None]
    nu_Dpi = nu_D@pi_ST
    Ds = init.resizeData(D,s,m)
    zeta_Dpi = nu_Dpi/np.sum(nu_Dpi,axis=-1)[...,None]
    
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
    vtf.writeVTK(D,imageValsS,imageNamesS,savedir+'testOutputiter' + str(itCurr) + '_D10.vtk',polyData=None)
    vtf.writeVTK(D,imageValsSpi,imageNamesSpi,savedir+'testOutputiter' + str(itCurr) + '_Dpi10.vtk',polyData=None)
    return
                            

###################################################################
# Optimization

def makePQ(S,nu_S,T,nu_T,norm=True):
    # initialize state vectors based on normalization 
    w_S = nu_S.sum(axis=-1)[...,None].type(dtype)
    w_T = nu_T.sum(axis=-1)[...,None].type(dtype)
    zeta_S = (nu_S/w_S).type(dtype)
    zeta_T = (nu_T/w_T).type(dtype)
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
    
    p0 = torch.cat((torch.zeros_like(q0),torch.sqrt(pi_STinit).clone().detach().flatten()),0).requires_grad_(True).type(dtype)
    
    return w_S,w_T,zeta_S,zeta_T,q0,p0,numS, Stilde, Ttilde, s, m, pi_STinit

def callOptimize(S,nu_S,T,nu_T,sigmaRKHS,sigmaVar,gamma,d,labs, savedir, its=100,kScale = torch.tensor(1.0).type(dtype),cA=1.0,cT=1.0,cS=10.0,cPi=1.0,beta=None):
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
    w_S, w_T,zeta_S,zeta_T,q0,p0,numS,Stilde,Ttilde,s,m, pi_STinit = makePQ(S,nu_S,T,nu_T)
    N = torch.tensor(S.shape[0]).type(dtype)
    s = s.cpu().numpy()
    m = m.cpu().numpy()

    pTilde = torch.zeros_like(p0).type(dtype)
    pTilde[0:numS] = kScale #torch.sqrt(kScale)*torch.sqrt(kScale)
    pTilde[numS:] = torch.tensor(1.0).type(dtype) #torch.sqrt(kScale)*1.0/(cScale*torch.sqrt(kScale))

    
    if (beta is None):
        # set beta to make ||mu_S - mu_T||^2 = 1
        if len(sigmaVar) == 1:
            Kinit = GaussLinKernelSingle(sig=sigmaVar[0],d=d,l=labs)
            cinit = Kinit(Ttilde,Ttilde,w_T*zeta_T,w_T*zeta_T).sum()
            k1 = Kinit(Stilde, Stilde, (w_S*zeta_S)@pi_STinit, (w_S*zeta_S)@pi_STinit)
            k2 = Kinit(Stilde, Ttilde, (w_S*zeta_S)@pi_STinit, w_T*zeta_T)
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
                k1 = Kinit(Stilde, Stilde, (w_S*zeta_S)@pi_STinit, (w_S*zeta_S)@pi_STinit).sum()
                k2 = -2.0*Kinit(Stilde, Ttilde, (w_S*zeta_S)@pi_STinit, w_T*zeta_T).sum()
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

    cst, dataloss = lossVarifoldNorm(Ttilde,w_T,zeta_T,zeta_S,GaussLinKernel(sigma=sigmaVar,d=d,l=labs,beta=beta),d,numS)
    piLoss = PiRegularizationSystem(zeta_S,nu_T,numS,d)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)

    loss = LDDMMloss(Kg,sigmaRKHS,d, numS, gamma, dataloss,piLoss,cA,cT,cPi)

    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    # keep track of both losses
    lossListH = []
    lossListDA = []
    lossListPI = []
    relLossList = []
    lossOnlyH = []
    lossOnlyDA = []
    def closure():
        optimizer.zero_grad()
        LH,LDA,LPI = loss(p0*pTilde, q0)
        L = LH+LDA+LPI
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())
        print("loss LPI ", LPI.detach().cpu().numpy())
        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))
        relLossList.append(np.copy(LDA.detach().cpu().numpy())/cst)
        lossListPI.append(np.copy(LPI.detach().cpu().numpy()))
        L.backward()
        return L
    
    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure) # default of 25 iterations in strong wolfe line search; will compute evals and iters until 25 unless reaches an optimum 
        osd = optimizer.state_dict()
        lossOnlyH.append(np.copy(osd['state'][0]['prev_loss']))
        if (np.mod(i,25) == 0):
            #p0Save = torch.clone(p0).detach()
            optimizer.zero_grad()
            printCurrentVariables(p0,i,Kg,sigmaRKHS,uCoeff,q0,d,numS,zeta_S,labs,s,m,savedir)
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")
        # save p0 for every 50th iteration and pi

    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListH),label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Varifold Norm, Final = {0:.6f}".format(lossListDA[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA)+np.asarray(lossListH),label="Total Cost, Final = {0:.6f}".format(lossListDA[-1]+lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListPI),label="Pi Cost, Final = {0:.6f}".format(lossListPI[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.legend()
    f.savefig(savedir + 'Cost.png',dpi=300)
    
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

    listpq = ShootingGrid(p0*pTilde,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,uCoeff,cA,cT)
    print("length of pq list is, ", len(listpq))
    Dlist = []
    nu_Dlist = []
    nu_DPilist = []
    Glist = []
    wGlist = []
    
    pi_STfinal = p0[(d+1)*numS:].detach().view(zeta_S.shape[-1],zeta_T.shape[-1])
    pi_STfinal = pi_STfinal**2
    pi_STfinal = pi_STfinal.cpu().numpy()
    print("pi final, ", pi_STfinal)
    f,ax = plt.subplots()
    im = ax.imshow(pi_STfinal)
    f.colorbar(im,ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Labels")
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
        # plot p0 as arrows
    listSp0 = np.zeros((numS*2,3))
    polyListSp0 = np.zeros((numS,3))
    polyListSp0[:,0] = 2
    polyListSp0[:,1] = np.arange(numS)# +1
    polyListSp0[:,2] = numS + np.arange(numS) #+ 1
    listSp0[0:numS,:] = S.detach().cpu().numpy()
    listSp0[numS:,:] = p0[numS:(d+1)*numS].detach().view(-1,d).cpu().numpy() + listSp0[0:numS,:]
    featsp0 = np.zeros((numS*2,1))
    featsp0[numS:,:] = p0[0:numS].detach().view(-1,1).cpu().numpy()
    vtf.writeVTK(listSp0,[featsp0],['p0_w'],savedir + 'testOutput_p0.vtk',polyData=polyListSp0)
    pNew = pTilde*p0
    A,tau = getATau(pNew[numS:(d+1)*numS].view(-1,d),q0[numS:].view(-1,d),q0[:numS].view(-1,1))
    np.savez(savedir + 'testOutput_values.npz',A0=A.detach().cpu().numpy(),tau0=tau.detach().cpu().numpy(),p0=p0.detach().cpu().numpy(),q0=q0.detach().cpu().numpy())    
    
    return Dlist, nu_Dlist, nu_DPilist, Glist, wGlist
