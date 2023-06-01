import numpy as np
import torch
from pykeops.torch import Vi,Vj
np_dtype = "float32" #"float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

import nibabel as nib


from crossModalityHamiltonianATSCalibrated import 
from saveState import loadParams, loadVariables

def resampleGauss(T,nu_T,D,sig):
    '''
    Resample particle information in T,nu_T onto set of particle locations at D using Gaussian with bandwith sig
    nu_D returned conserves mass
    '''
    
    T_i = Vi(torch.tensor(T).type(dtype))
    D_j = Vj(torch.tensor(D).type(dtype))

    D2 = T_i.sqdist(D_j)
    K = (-D2 / (2.0*sig*sig)).exp() # T points by D points 
    
    # \sum_{F} nu_T(f)
    nu_i = Vi(torch.tensor(nu_T).type(dtype))
    nu_j = Vj(torch.ones((D.shape[0],nu_T.shape[-1])).type(dtype))
    
    K = K * (nu_i*nu_j).sum()
    nu_TD = K.sum_reduction(dim=0)
    print("nu_TD shape should be same as D, ", nu_TD.shape)
    print("D shape, ", D.shape)
    
    # normalize
    nu_TD = np.squeeze(nu_TD.cpu().numpy())
    w_T = np.sum(nu_T)
    w_D = np.sum(nu_TD)
    
    return nu_TD*(w_T/w_D)
    

def makeGrid(T,nu_T,res,savedir,dimE=2):
    '''
    Makes grid covering support of T with all weights of 1
    res is scalar, used in the same way for all dimensions
    '''
    
    ranges = np.max(T,axis=0) - np.min(T,axis=0)
    numCubes = np.ceil(ranges/res)
    
    if numCubes[-1] == 0 or dimE == 2:
        dimEff = 2
        numCubes = numCubes[0:2] + 5
    else:
        dimEff = 3
        numCubes = numCubes + 5 # outline support by 5 extra voxels in each dimension
        
    x0 = np.arange(numCubes[0])*res + (np.min(T[:,0]) - 5*res)
    x1 = np.arange(numCubes[1])*res + (np.min(T[:,1]) - 5*res)
    
    if (dimEff == 3):
        x2 = np.arange(numCubes[2])*res + (np.min(T[:,2]) - 5*res)
    else:
        x2 = np.arange(1) + np.mean(T[:,-1])
    X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
    X = np.stack((np.ravel(X0),np.ravel(X1),np.ravel(X2)),axis=-1)
         
    nu_X = resampleGauss(T,nu_T,X,res)
    nu_Xgrid = np.reshape(nu_X,(X0.shape[0],X0.shape[1],X0.shape[2],nu_X.shape[-1]))
    Xgrid = np.stack((X0,X1,X2),axis=-1)
    
    empty_header = nib.Nifti1Header()
    wIm = nib.Nifti1Image(np.sum(nu_Xgrid,axis=-1), np.eye(4), empty_header)
    mIm = nib.Nifti1Image((np.argmax(nu_Xgrid,axis=-1) + 1)*(np.sum(nu_Xgrid,axis=-1)>0),np.eye(4),empty_header)
    nib.save(wIm,savedir + 'nu_Tsum_grid.nii.gz')
    nib.save(mIm,savedir + 'nu_Tmax_grid.nii.gz')
    
    return X,nu_X,Xgrid,nu_Xgrid
    
def shootFromP0():
    '''
    Load variables and parameters from filenames
    Use shootinggrid to reshoot new set of particles --> check if need q0?
    '''

    
        