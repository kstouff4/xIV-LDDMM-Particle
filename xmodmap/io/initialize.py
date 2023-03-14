import numpy as np
from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
dtype = torch.cuda.FloatTensor 

import nibabel as nib
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def applyAffine(Z, nu_Z, A, tau):
    '''
    Makes a new set of particles based on an input set and applying the affine transformation given by matrix A and translation, tau
    '''
    R = torch.clone(Z)
    nu_R = torch.clone(nu_Z)
    R = R@A.T + tau
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