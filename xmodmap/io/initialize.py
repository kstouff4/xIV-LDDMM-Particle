import numpy as np
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
