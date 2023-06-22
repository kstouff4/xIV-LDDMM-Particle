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

sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import nibabel as nib

#################################################################################
# Functions for initializing images
def alignImage(imgFile,segFile,axesOrder,flip):
    im = nib.load(imgFile)
    imData = np.squeeze(im.dataobj)
    seg = nib.load(segFile)
    segData = np.squeeze(seg.dataobj)
    
    for f in range(len(flip)):
        if flip[f] == True:
            imData = np.flip(imData,axis=f)
            segData = np.flip(segData,axis=f)
    imNew = np.transpose(imData,axes=axesOrder)
    segNew = np.transpose(segData,axes=axesOrder)
    
    imgNewn = nib.Nifti1Image(imNew, im.affine)
    segNewn = nib.Nifti1Image(segNew, seg.affine)
    nib.save(imgNewn,imgFile.replace(imgFile.split('.')[-1],'_rotated.nii.gz'))
    nib.save(segNewn,segFile.replace(segFile.split('.')[-1],'_rotated.nii.gz'))
    return
    
################################################################################
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

def getFinalTransforms(p0,Slist,nuSlist,lamb,numVars=6,numTrans=3):
    ASlist = [Slist[0]]
    As = []
    ts = []
    angles = []
    for i in range(numTrans):
        p = p0[i*numVars:(i+1)*numVars]
        A = init.get3DRotMatrix(p[0],p[1],p[2])
        As.append(A)
        angles.append(p[0:3])
        ts.append(p[3:]*lamb)
        AS,nuAS = init.applyAffine(Slist[i+1],nuSlist[i+1],A,p[3:]*lamb,bc=False)
        ASlist.append(AS)
    return ASlist, As, ts, angles
        

# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(K,nuSant,nuSpost):
    # used to have nuSCat
    antNu = torch.cat(nuSant)
    postNu = torch.cat(nuSpost)

    def loss(ASant,ASpost):
        # sS will be in the form of q (w_S,S,x_c)
        Sant = torch.cat(ASant)
        Spost = torch.cat(ASpost)
        print("comparing sizes in varifold norm")
        print(Spost.detach().shape)
        print(Sant.detach().shape)
        print(postNu.detach().shape)
        print(antNu.detach().shape)
        k = K(Sant,Spost,antNu,postNu)
        k2 = K(Sant,Sant,antNu,antNu)
        k3 = K(Spost,Spost,postNu,postNu)
        
        return k3.sum() + k2.sum() - 2.0*k.sum()

    return loss

'''
def lossConstraints(ASlist,nuSlist,zCoords):
    cost = torch.tensor(0.0).type(dtype)
    print("lengths ", len(ASlist))
    print(len(nuSlist))
    for i in range(len(ASlist)):
        S = ASlist[i]
        nuS = nuSlist[i]
        for z in range(len(zCoords)):
            cost += ( nuS[:,z]*(S[:,-1] - zCoords[z])**2).sum()
    return cost
'''

def LDDMMloss(gammaR,dataloss,Sant,Spost,nuSant,nuSpost,lamb,numVars=6):
    #ASlist.append(Slist[0]) # assume first block doesn't move
    def loss(p0):
        pCost = 0
        ASant = []
        ASpost = [Spost[0]]
        for i in range(len(Sant)-1):
            p = p0[i*numVars:(i+1)*numVars]
            # assume first posterior interface is always fixed 
            z,_ = transform(Spost[i+1],nuSpost[i+1],p[0:3],p[3:]*lamb)
            ASpost.append(z)
            z,_ = transform(Sant[i],nuSant[i],p[0:3],p[3:]*lamb)
            ASant.append(z)
            pCost += (p*p).sum()
        p = p0[-numVars:]
        z,_ = transform(Sant[-1],nuSant[-1],p[0:3],p[3:]*lamb)
        ASant.append(z)
        return (gammaR/2.0 * pCost), dataloss(ASant,ASpost)

    return loss

############################################################################

def makePQ(Slist,nuSlist,numParams=6,numTrans=3):
    # group features into anterior vs. posterior; assume blocks are ordered anterior to posterior
    # default to 6 parameters per block other than first (number of blocks-1)
    Spost = [Slist[0]]
    nuSpost = [nuSlist[0]]
    mi = torch.min(torch.cat(Slist),axis=0).values
    ma = torch.max(torch.cat(Slist),axis=0).values
    s = 1.0/torch.max(ma-mi)
    print("lambda is, ", s.detach())
    f = torch.argmax(nuSlist[0][0],axis=-1)
    Sant = []
    nuSant = []
    for i in range(1,len(Slist)-1):
        n = nuSlist[i]
        inds = torch.argmax(n,axis=-1) == f
        nuSant.append(n[inds,:])
        Sant.append(Slist[i][inds,:])
        inds = torch.argmax(n,axis=-1) != f
        nuSpost.append(n[inds,:])
        Spost.append(Slist[i][inds,:])
        
        f = torch.argmax(nuSpost[i][0],axis=-1)
    Sant.append(Slist[-1])
    nuSant.append(nuSlist[-1])
    
    print("lengths of posterior list and anterior list should be the same")
    print(len(Spost))
    print(len(Sant))
    p0 = torch.zeros((numParams*numTrans)).type(dtype).requires_grad_(True)
    print("size of p0 is", p0.detach().cpu().numpy().shape)
    return p0, Spost,nuSpost,Sant,nuSant,s


def callOptimize(Slist,nu_Slist,sigmaVar,gammaR,savedir,its=100,d=3,numVars=6):
    '''
    Parameters:
        Slist, nu_Slist = image varifold (blocks of MRI)
        sigmaVar = list of sigmas for varifold norm (assumed Gaussian)
        gamma = weight of regularization terms vs matching 
        d = dimensions of space
        savedir = location to save cost graphs and p0 in
        its = iterations of LBFGS steps (each step has max 10 iterations and 15 evaluations of objective function)
    '''
    p0,Spost,nuSpost,Sant,nuSant,lamb = makePQ(Slist,nu_Slist,numVars,len(Slist)-1)

    labs = nu_Slist[0].shape[-1]
    
    # set beta to make ||mu_S - mu_T||^2 = 1
    if len(sigmaVar) == 1:
        Kinit = GaussLinKernelSingle(sig=sigmaVar[0],d=d,l=labs)
        cinit = Kinit(torch.cat(Spost),torch.cat(Sant),torch.cat(nuSpost),torch.cat(nuSant)).sum()
        beta = torch.tensor(2.0/(cinit)).type(dtype)
        print("beta is ", beta.detach().cpu().numpy())
        beta = [(0.6/sigmaVar[0])*torch.clone(2.0/(cinit)).type(dtype)] 
        
    # print out indiviual costs
    else:
        print("different varifold norm at beginning")
        beta = []
        for sig in sigmaVar:
            print("sig is ", sig.detach().cpu().numpy())
            Kinit = GaussLinKernelSingle(sig=sig,d=d,l=labs)
            cinit = Kinit(torch.cat(Spost),torch.cat(Sant),torch.cat(nuSpost),torch.cat(nuSant)).sum()
            beta.append((0.6/sig)*torch.clone(2.0/(cinit)).type(dtype))

    dataloss = lossVarifoldNorm(GaussLinKernel(sigma=sigmaVar,d=d,l=labs,beta=beta),nuSant,nuSpost)

    loss = LDDMMloss(gammaR,dataloss,Sant,Spost,nuSant,nuSpost,lamb,numVars=6)

    optimizer = torch.optim.LBFGS([p0], max_eval=15, max_iter=10,line_search_fn = 'strong_wolfe',history_size=100,tolerance_grad=1e-8,tolerance_change=1e-10)
    print("performing optimization...")
    start = time.time()
    
    # keep track of both losses
    lossListH = []
    lossListDA = []
    lossOnlyH = []
    lossOnlyDA = []

    def closure():
        optimizer.zero_grad()
        LH,LDA = loss(p0)
        L = LH+LDA
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())

        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))

        L.backward()
        return L
    
    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure)
        print("state of optimizer")
        osd = optimizer.state_dict()
        #print(osd)
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

    ASlist, As, ts, angles = getFinalTransforms(p0,Slist,nu_Slist,lamb)
    Asnp = []
    tsnp = []
    anglesnp = []
    ASlistnp = []
    for i in range(len(As)):
        Asnp.append(As[i].detach().cpu().numpy())
        tsnp.append(ts[i].detach().cpu().numpy())
        anglesnp.append(angles[i].detach().cpu().numpy())
        ASlistnp.append(ASlist[i].detach().cpu().numpy())
    ASlistnp.append(ASlist[-1].detach().cpu().numpy())
    np.savez(savedir + 'testOutput_values.npz',ASlist=ASlistnp,ts=tsnp,As=Asnp,angles=anglesnp)
    return ASlistnp,Asnp,tsnp

