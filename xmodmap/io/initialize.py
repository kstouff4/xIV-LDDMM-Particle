import numpy as np
import scipy as sp
from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
dtype = torch.cuda.FloatTensor #DoubleTensor

import nibabel as nib
import pandas as pd



def applyAffine(Z, nu_Z, A, tau):
    '''
    Makes a new set of particles based on an input set and applying the affine transformation given by matrix A and translation, tau
    '''
    print("max before ", torch.max(Z,axis=0))
    R = torch.clone(Z)
    nu_R = torch.clone(nu_Z)
    R = R@A.T + tau
    print("max ", torch.max(R,axis=0))
    return R,nu_R

def alignBaryCenters(S,nu_S,T,nu_T):
    '''
    Translate S to be on barycenter of T with barycenters computed based on sum over features in nu_S and nu_T
    '''
    wS = nu_S.sum(axis=-1)
    wT = nu_T.sum(axis=-1)
    
    xcS = (S*wS).sum(dim=0)/(wS.sum(dim=0))
    xcT = (T*wT).sum(dim=0)/(wT.sum(dim=0))
    
    tau = xcT - xcS
    Snew = S + tau
    
    return Snew

def get3DRotMatrix(thetaX,thetaY,thetaZ):
    '''
    thetaX, thetaY, and thetaZ should all be torch tensors in radians 
    '''
    Ax = torch.zeros((3,3)).type(dtype)
    Ax[0,0] = torch.tensor(1.0)
    Ax[1,1] = torch.cos(thetaX)
    Ax[2,2] = Ax[1,1]
    Ax[1,2] = -torch.sin(thetaX)
    Ax[2,1] = torch.sin(thetaX)
    
    Ay = torch.zeros((3,3)).type(dtype)
    Ay[0,0] = torch.cos(thetaY)
    Ay[0,2] = torch.sin(thetaY)
    Ay[2,0] = -torch.sin(thetaY)
    Ay[2,2] = torch.cos(thetaY)
    Ay[1,1] = torch.tensor(1.0)
    
    Az = torch.zeros((3,3)).type(dtype)
    Az[0,0] = torch.cos(thetaZ)
    Az[0,1] = -torch.sin(thetaZ)
    Az[1,0] = torch.sin(thetaZ)
    Az[1,1] = torch.cos(thetaZ)
    Az[2,2] = torch.tensor(1.0)
    
    R = Ax@Ay@Az
    print("R: ", R)
    return R

def get3DRotMatrixAxis(theta,ax=None):
    # ax = vector on unit sphere (e.g. 
    # theta is numpy number in radians
    if (ax is None):
        ax = np.zeros((3,1))
        ax[1] = 1.0
    K = np.zeros((3,3))
    K[0,1] = -ax[2]
    K[0,2] = ax[1]
    K[1,0] = ax[2]
    K[2,0] = -ax[1]
    K[1,2] = -ax[0]
    K[2,1] = ax[0]
    #A = np.cross(-theta*ax,np.identity(3),axisa=0,axisb=0)
    R = sp.linalg.expm(theta*K)
    return torch.from_numpy(R).type(dtype)

def rescaleData(S,T):
    '''
    Rescales Data to be in bounding box of [0,1]^d
    Returns rescaled data and rescaling coefficient (one coefficient for each direction)
    '''
    
    X = torch.cat((S,T))
    m = torch.min(X,axis=0).values
    rang = torch.max(X,axis=0).values - m
    s = torch.max(rang)
    Stilde = (S - m)*(1.0/s)
    Ttilde = (T-m)*(1.0/s)
    
    return Stilde,Ttilde,s,m

def resizeData(Xtilde,s,m):
    '''
    Inverse of rescaleData. Takes coefficient of scaling s and rescales it appropriately.
    '''
    X = Xtilde*s + m
    return X

def scaleDataByVolumes(S,nuS,T,nuT,dRel=3):
    '''
    Scale the source by the ratio of the overall volumes. Assume isotropic scaling.
    '''
    # center source and target at 0,0
    minS = torch.min(S,axis=0).values
    maxS = torch.max(S,axis=0).values
    print("minS size", minS.shape)
    Sn = S - torch.tensor(0.5)*(maxS+minS)
    if (dRel < 3):
        vS = torch.prod(maxS[:dRel] - minS[:dRel])
        vT = torch.prod(torch.max(T,axis=0).values[:dRel] - torch.min(T,axis=0).values[:dRel])
        print("vT versus vS")
        print(vT)
        print(vS)
        scaleF = vT/vS
        scaleF = scaleF**(1.0/dRel)
        Sn[:,0:dRel] = Sn[:,0:dRel]*scaleF
        nuSn = nuS*(scaleF**dRel)
    else:
        vS = torch.prod(maxS - minS)
        vT = torch.prod(torch.max(T,axis=0).values - torch.min(T,axis=0).values)
        scaleF = vT/vS
        scaleF = scaleF**(1.0/3.0)
        Sn = scaleF*Sn
        nuSn = nuS*(scaleF**3)
    print("scale factor is, ", scaleF)
    
    return Sn,nuSn
