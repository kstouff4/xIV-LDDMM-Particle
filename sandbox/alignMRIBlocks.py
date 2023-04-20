import os
import time
import numpy as np
from numpy import random

import torch
from torch.autograd import grad

import pykeops
#import socket
#pykeops.set_build_folder("~/.cache/keop" + pykeops.__version__ + "_" + (socket.gethostname()))

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

# Varifold Norm Kernel assuming Gaussian for spatial kernel and Euclidean for features
# Multiple scales allowed with sigma 
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

# |\mu_s - \mu_T |_sigma^2
def GaussLinKernelSingle(sig,d,l):
    # u and v are the feature vectors 
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    K = (-D2 / (2.0*sig*sig)).exp() * (u * v).sum()
    return K.sum_reduction(axis=1)

################################################################################
def transform(Z,nu_Z,angles,t):
    A = init.get3DRotMatrix(angles[0],angles[1],angles[2])
    Zn,nu_Zn = init.applyAffine(Z,nu_Z,A,t,bc=False)
    return Zn, nu_Zn

def getFinalTransforms(p0,Slist,nuSlist):
    ASlist = []
    As = []
    ts = []
    angles = []
    for i in range(len(p0)):
        p = p0[i]
        A = init.get3DRotMatrix(p[0],p[1],p[2])
        As.append(A)
        angles.append(p[0:3])
        ts.append(p[3:])
        AS,nuAS = init.applyAffine(Slist[i],nuSlist[i],A,p[3:],bc=False)
        ASlist.append(AS)
    return ASlist, As, ts, angles
        

# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(K,nuSCat):

    def loss(ASlist):
        # sS will be in the form of q (w_S,S,x_c)
        Scat = torch.cat((ASlist[0],ASlist[1]))
        for i in range(2,len(ASlist)):
            Scat = torch.cat((Scat,ASlist[i]))
        k = K(Scat,Scat,nuScat,nuSCat)
        return k.sum()

    return loss

def lossConstraints(ASlist,nuSlist,zCoords):
    cost = torch.tensor(0.0).type(dtype)
    for i in range(len(ASlist)):
        S = ASlist[i]
        nuS = nuSlist[i]
        for z in range(len(zCoords)):
            cost += ( nuS[:,z]*(S[:,-1] - zCoords[z])**2).sum()
    return cost

def LDDMMloss(gammaR,gammaC,dataloss,Slist,nuSlist,zCoords):
    ASlist = []
    #ASlist.append(Slist[0]) # assume first block doesn't move
    pCost = torch.tensor(0.0).type(dtype)
    def loss(p0):
        for i in range(len(p0)):
            p = p0[i]
            z,_ = transform(Slist[i],nuSlist[i],p[0:3],p[3:])
            ASlist.append(z)
            pCost += (p*p).sum()
        return (gammaR/2.0 * pCost), dataloss(ASlist),(gammaC/2.0)*lossConstraints(ASlist,nuSlist,zCoords)

    return loss

############################################################################

def makePQ(Slist,numParams=6):
    # initialize state vectors based on normalization 
    # default to 6 parameters per block (number of blocks)
    Stilde, s, m = init.rescaleData(Slist)
    
    p0 = []
    for i in range(len(Slist)):
        p0.append(torch.zeros((numParams)).requires_grad_(True).type(dtype))
    
    return p0, Stilde, s, m


def callOptimize(Slist,nu_Slist,sigmaVar,gammaR,gammaC,savedir, zCoordsO,its=100,d=3,numVars=6):
    '''
    Parameters:
        Slist, nu_Slist = image varifold (blocks of MRI)
        sigmaVar = list of sigmas for varifold norm (assumed Gaussian)
        gamma = weight of regularization terms vs matching 
        d = dimensions of space
        savedir = location to save cost graphs and p0 in
        its = iterations of LBFGS steps (each step has max 10 iterations and 15 evaluations of objective function)
    '''
    p0,Stilde,s,m = makePQ(Slist,numVars)
    zCoords = (zCoordsO-m)*(1.0/s)
    #N = torch.tensor(S.shape[0]).type(dtype)
    s = s.cpu().numpy()
    m = m.cpu().numpy()
    labs = nu_Slist[0].shape[-1]
    
    StildeCat = torch.cat(torch.clone(Stilde[0]),torch.clone(Stilde[1])).type(dtype)
    nu_SCat = torch.cat(torch.clone(nu_Slist[0]),torch.clone(nu_Slist[1])).type(dtype)
    for i in range(2,len(Stilde)):
        StildeCat = torch.cat(StildeCat,torch.clone(Stilde[i])).type(dtype)
        nu_SCat = torch.cat(nu_SCat,torch.clone(nu_SCat[i])).type(dtype)
    
    # set beta to make ||mu_S - mu_T||^2 = 1
    if len(sigmaVar) == 1:
        Kinit = GaussLinKernelSingle(sig=sigmaVar[0],d=d,l=labs)
        cinit = Kinit(StildeCat,StildeCat,nu_SCat,nu_SCat).sum()
        beta = torch.tensor(2.0/(cinit)).type(dtype)
        print("beta is ", beta.detach().cpu().numpy())
        beta = [torch.clone(2.0/(cinit)).type(dtype)] 
        
    # print out indiviual costs
    else:
        print("different varifold norm at beginning")
        beta = []
        for sig in sigmaVar:
            print("sig is ", sig.detach().cpu().numpy())
            Kinit = GaussLinKernelSingle(sig=sig,d=d,l=labs)
            cinit = Kinit(StildeCat,StildeCat,nu_SCat,nu_SCat).sum()
            beta.append(torch.clone(2.0/(cinit)).type(dtype))

    dataloss = lossVarifoldNorm(GaussLinKernel(sigma=sigmaVar,d=d,l=labs,beta=beta),nu_SCat)

    loss = LDDMMloss(gammaR,gammaC, dataloss,Stilde,nu_Slist,zCoords)

    optimizer = torch.optim.LBFGS(p0, max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    # keep track of both losses
    lossListH = []
    lossListDA = []
    lossOnlyH = []
    lossOnlyDA = []
    lossListC = []
    def closure():
        optimizer.zero_grad()
        LH,LDA,LC = loss(p0)
        L = LH+LDA+LC
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())
        print("loss LC ", LC.detach().cpu().numpy())
        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))
        lossListC.append(np.copy(LC.detach().cpu().numpy()))
        L.backward()
        return L
    
    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure) # default of 25 iterations in strong wolfe line search; will compute evals and iters until 25 unless reaches an optimum 
        print("state of optimizer")
        osd = optimizer.state_dict()
        print(osd)

        lossOnlyH.append(np.copy(osd['state'][0]['prev_loss']))
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")
    
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListH),label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Varifold Norm, Final = {0:.6f}".format(lossListDA[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA)+np.asarray(lossListH),label="Total Cost, Final = {0:.6f}".format(lossListDA[-1]+lossListH[-1]))
    ax.plot(np.arange(len(lossListH)),np.asarray(lossListDA),label="Constraint Cost, Final = {0:.6f}".format(lossListC[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + 'Cost.png',dpi=300)

    ASlist, As, ts, angles = getFinalTransforms(p0,Stilde,nu_Slist)
    ASlistScale = []
    for St in Alist:
        ASlistScale.append(init.resizeData(St,s,m))
    Asnp = []
    tsnp = []
    anglesnp = []
    ASlistnp = []
    ASlistScalenp = []
    for i in range(len(ASlist)):
        Asnp.append(As[i].detach().cpu().numpy())
        tsnp.append(ts[i].detach().cpu().numpy())
        anglesnp.append(angles[i].detach().cpu().numpy())
        ASlistnp.append(ASlist[i].detach().cpu().numpy())
        ASlistScalenp.append(ASlistScale[i].detach().cpu().numpy())
    np.savez(savedir + 'testOutput_values.npz',ASlist=ASlistnp,ts=tsnp,As=Asnp,angles=anglesnp,ASlistScale=ASlistScalenp,s=s,m=m)
    return ASlistScalenp,Asnp,tsnp,s,m

