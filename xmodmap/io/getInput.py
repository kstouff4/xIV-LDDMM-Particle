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

def readFromPrevious(npzFile):
    '''
    Assumes source deformed to target previously.
    Initializes new source as deformed source.
    '''
    npz = np.load(npzFile)
    S = torch.tensor(npz['D']).type(dtype)
    nu_S = torch.tensor(npz['nu_D']).type(dtype)
    T = torch.tensor(npz['T']).type(dtype)
    nu_T = torch.tensor(npz['nu_T']).type(dtype)
    
    return S,nu_S,T,nu_T

def makeFromSingleChannelImage(imageFile,resXYZ,bg=0,ordering=None,ds=1):
    '''
    Makes discrete particle representation from image file (NIFTI or ANALYZE).
    Assumes background has value 0 and excluded as no data.
    
    Centers particles around the origin based on bounding box of coordinates.
    '''
    
    imInfo = nib.load(imageFile)
    im = np.squeeze(np.asanyarray(imInfo.dataobj)).astype('float32')
    dims = im.shape
    if (ds > 1):
        if len(dims) == 2:
            im = im[0::ds,0::ds]
        elif len(dims) == 3:
            im = im[0::ds,0::ds,0::ds]
    dims = im.shape
    print("dims is ", dims)
    x0 = np.arange(dims[0])*resXYZ
    x0 -= np.mean(x0)
    x1 = np.arange(dims[1])*resXYZ
    x1 -= np.mean(x1)
    if len(dims) > 2:
        x2 = np.arange(dims[2])*resXYZ
        x2 -= np.mean(x2)
    else:
        x2 = np.zeros((1,1)) # default to centering 2d image at 0
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0).type(dtype),torch.tensor(x1).type(dtype),torch.tensor(x2).type(dtype),indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    print("size of S:", S.shape)
    listOfNu = []
    
    if (ordering is not None):
        uniqueVals = ordering
    else:
        uniqueVals = np.unique(im)
        if (bg is not None):
            uniqueVals = uniqueVals[uniqueVals != bg]
        
    numUnique = len(uniqueVals)

    keepSum = torch.zeros((S.shape[0],1)).type(dtype)
    for u in range(len(uniqueVals)):
        n = torch.tensor((im == uniqueVals[u])).type(dtype)
        listOfNu.append(n.flatten())
        print("n flatten shape ", n.flatten().shape)
        keepSum += n.flatten()[...,None]
    toKeep = torch.squeeze(keepSum > 0)
    print("toKeep shape ", toKeep.shape)
    listOfNewNu = []
    print("length of list ", len(listOfNu))
    for l in listOfNu:
        print("l shape ", l[toKeep].shape)
        listOfNewNu.append(l[toKeep])
    nu_S = torch.stack(listOfNewNu,axis=-1).type(dtype)

    #toKeep = nu_S.sum(axis=-1) > 0
    S = S[toKeep]
    #nu_S = nu_S[toKeep]

    return S,nu_S

def readParticleApproximation(particleNPZ):
    '''
    If in the form of particle approximation, then spatial coordinates will be saved as "Z" and features as "nu_Z"
    '''
    npz = np.load(particleNPZ)
    S = torch.tensor(npz['Z']).type(dtype)
    nu_S = torch.tensor(npz['nu_Z']).type(dtype)

    return S,nu_S

def readSpaceFeatureCSV(coordCSV,coordNames,featCSV,featNames,scale=None,labs=None):
    '''
    For reading in a csv with each row representative of measure (e.g. cell or single mRNA)
    Scale datapoints to mm if in different coordinates (e.g. microns --> scale = 1e-3)
    Center data points around 0,0
    '''
    df_s = pd.read_csv(coordCSV)
    df_f = pd.read_csv(featCSV)
    if (len(featNames) > 1):
        nu_S = torch.tensor(df_f[featNames].values).type(dtype)
    elif labs is not None:
        listOfNu = []
        nu_S_single = df_f[featNames].values
        if (np.min(nu_S_single) == 1):
            nu_S_single -= 1
        for u in range(labs):
            n = torch.tensor((nu_S_single == u)).type(dtype)
            listOfNu.append(n.flatten())
        nu_S = torch.stack(listOfNu,axis=-1).type(dtype)
    else:
        nu_S_single = df_f[featNames].values
        uVals = np.unique(nu_S_single)
        listOfNu = []
        for u in uVals:
            n = torch.tensor((nu_S_single == u)).type(dtype)
            listOfNu.append(n.flatten())
        nu_S = torch.stack(listOfNu,axis=-1).type(dtype)
        
    S = torch.tensor(df_s[coordNames].values).type(dtype)
    
    if scale is not None:
        S = torch.tensor(scale*df_s[coordNames].values).type(dtype)
    S = S - torch.mean(S,axis=0)
    if (S.shape[-1] < 3):
        S = torch.cat((S,S[...,0][...,None]*0),-1).type(dtype)
    return S,nu_S
    