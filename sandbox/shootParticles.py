import numpy as np
import torch
from pykeops.torch import Vi,Vj
np_dtype = "float32" #"float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

import nibabel as nib


from crossModalityHamiltonianATSCalibrated import GaussKernelHamiltonian, ShootingGrid
from saveState import loadParams, loadVariables
import sys
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../xmodmap/')
sys_path.append('../xmodmap/io/')
import initialize as init
import getOutput as gO

def resampleGauss(T,nu_T,D,sig):
    '''
    Resample particle information in T,nu_T onto set of particle locations at D using Gaussian with bandwith sig
    nu_D returned conserves mass
    '''
    print("T shape and D shape")
    print(T.shape)
    print(D.shape)
    T_i = Vi(torch.tensor(T).type(dtype))
    D_j = Vj(torch.tensor(D).type(dtype))

    D2 = T_i.sqdist(D_j)
    K = (-D2 / (2.0*sig*sig)).exp() # T points by D points 
    bw = 30
    # \sum_{F} nu_T(f)
    if (nu_T.shape[-1] > bw):
        nu_TD = torch.zeros((D.shape[0],nu_T.shape[-1])).type(dtype)
        count = 0
        for i in range(0,nu_T.shape[-1] - bw,bw):
            nu_i = Vi(torch.tensor(nu_T[:,i:i+bw]).type(dtype))
            nu_j = Vj(torch.ones((D.shape[0],bw)).type(dtype))
            N = (nu_i*nu_j)
            K1 = K*N
            nu_TD[:,i:i+bw] = K1.sum_reduction(dim=0)
            count = i
        count = count+bw
        nu_i = Vi(torch.tensor(nu_T[:,count:]).type(dtype))
        nu_j = Vj(torch.ones((D.shape[0],nu_T[:,count:].shape[-1])).type(dtype))
        N = (nu_i*nu_j)
        K1 = K*N
        nu_TD[:,count:] = K1.sum_reduction(dim=0)                
        
    else:
        nu_i = Vi(torch.tensor(nu_T).type(dtype))
        print("nui shape, ", nu_i.shape)
        nu_j = Vj(torch.ones((D.shape[0],nu_T.shape[-1])).type(dtype))
        print("nuj shape, ", nu_j.shape)
        print("K shape, ", K.shape)
        N = (nu_i*nu_j)
        print("N shape, ", N.shape)
        K = K * N
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
        numCubes = numCubes[0:2] + 20
    else:
        dimEff = 3
        numCubes = numCubes + 20 # outline support by 5 extra voxels in each dimension
        
    x0 = np.arange(numCubes[0])*res + (np.min(T[:,0]) - 10*res)
    x1 = np.arange(numCubes[1])*res + (np.min(T[:,1]) - 10*res)
    
    if (dimEff == 3):
        x2 = np.arange(numCubes[2])*res + (np.min(T[:,2]) - 10*res)
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
    
def shootBackwards(paramFile,variableFile,Z,w_Z,dimE=3):
    '''
    Load variables and parameters from filenames
    Use shootinggrid to reshoot new set of particles --> check if need q0?
    '''
    uCoeff,sigmaRKHS,sigmaVar,beta,d,labs,numS,pTilde,gamma,cA,cT,cPi,single = loadParams(paramFile)
    q0,p0,Ttilde,w_T,s,m = loadVariables(variableFile)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)
    
    x = torch.arange(3).type(dtype)
    XG,YG,ZG = torch.meshgrid((x,x,x),indexing='ij')
    qGrid = torch.stack((XG.flatten(),YG.flatten(),ZG.flatten()),axis=-1).type(dtype)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG)).type(dtype) 

    listpq = ShootingGrid(p0,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,uCoeff,cA,cT,dimE,single=single,T=Z.flatten(),wT=w_Z.flatten())
    
    Zlist = []
    wZlist = []
    
    for t in range(len(listpq)):
        Tt = listpq[t][4]
        Zlist.append(init.resizeData(Tt.detach().view(-1,d).cpu().numpy(),s,m))
        wTt = listpq[t][5]
        wZlist.append(wTt.detach().cpu().numpy())
    
    return Zlist[-1],wZlist[-1]

def shootForwards(paramFile,variableFile,Q,w_Q,dimEff=3):
    uCoeff,sigmaRKHS,sigmaVar,beta,d,labs,numS,pTilde,gamma,cA,cT,cPi,single = loadParams(paramFile)
    q0,p0,Ttilde,w_T,s,m = loadVariables(variableFile)
    
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)
    #p0.requires_grad=True
    #q0.requires_grad=True
    listpq = ShootingGrid(p0,q0,Q,w_Q,Kg,sigmaRKHS,d,numS,uCoeff,cA,cT,dimEff,single=single)
    
    Zlist = []
    wZlist = []
    
    for t in range(len(listpq)):
        Tt = listpq[t][2]
        Zlist.append(init.resizeData(Tt.detach().view(-1,d).cpu().numpy(),s,m))
        wTt = listpq[t][3]
        wZlist.append(wTt.detach().cpu().numpy())
    
    return Zlist[-1],wZlist[-1]

def shootTargetGridSlice(Ttilde,nu_T,index,res,savedir,paramFile,variableFile):
    '''
    Returns grid rendering (X x Y x Z x 3) and similar for features covering the support of slice index in T dataset 
    input in tensors
    '''
    u,i = torch.unique(Ttilde[:,-1],return_inverse=True)
    
    # make grid covering 1 slice 
    if (index >= 0):
        T = Ttilde[i == index,...]
        nuT = nu_T[i == index,...]
        print("ranges of Ttilde slice")
        print(torch.min(T,axis=0))
        print(torch.max(T,axis=0))
        X,nu_X,Xgrid,nu_Xgrid = makeGrid(T.cpu().numpy(),nuT.cpu().numpy(),res,savedir + 'Slice_' + str(index) + '_',dimE=2)
        print("ranges of Ttilde grid slice")
        print(np.min(X,axis=0))
        print(np.max(X,axis=0))

        w_X = np.sum(nu_X,axis=-1)
        Xs,w_Xs = shootBackwards(paramFile,variableFile,torch.tensor(X).type(dtype),torch.tensor(w_X).type(dtype),dimE=3)

        nu_Xs = (nu_X/np.sum(nu_X,axis=-1)[...,None])*w_Xs[...,None]

        Xsgrid = np.reshape(Xs,Xgrid.shape)
        nu_Xsgrid = np.reshape(nu_Xs,nu_Xgrid.shape)
        np.savez(savedir + 'Slice_' + str(index) + '_originalAndShot.npz',Xs=Xs,w_Xs=w_Xs,X=X,nu_X=nu_X,Xsgrid=Xsgrid,nu_Xsgrid=nu_Xsgrid,Xgrid=Xgrid,nu_Xgrid=nu_Xgrid)
    
    return Xsgrid, nu_Xsgrid, Xs, nu_Xs 

def constructVelocityFields(paramFile,variableFile,outpath,dimEff=3):
    '''
    Defaults to constructing grid to cover source and target with smallest scale RKHS
    '''
    uCoeff,sigmaRKHS,sigmaVar,beta,d,labs,numS,pTilde,gamma,cA,cT,cPi,single = loadParams(paramFile)
    q0,p0,Ttilde,w_T,s,m = loadVariables(variableFile)

    res = sigmaRKHS[-1]
    
    # backwards velocity field (from target to source)
    maT = torch.max(Ttilde,axis=0).values
    miT = torch.min(Ttilde,axis=0).values
    print("max values, ", maT)
    x0T = torch.arange(miT[0] - 2.0*res, maT[0] + 2.0*res, res)
    x1T = torch.arange(miT[1] - 2.0*res, maT[1] + 2.0*res, res)
    x2T = torch.arange(miT[2] - 2.0*res, maT[2] + 2.0*res, res)
    X0T, X1T, X2T = torch.meshgrid((x0T,x1T,x2T),indexing='ij')
    XT = torch.stack((X0T.flatten(),X1T.flatten(),X2T.flatten()),axis=-1).type(dtype)
    numG = XT.shape[0]
    qGrid = XT.flatten()
    qGridw = torch.ones((numG)).type(dtype)    

    vT,wT = shootBackwards(paramFile,variableFile,qGrid,qGridw)
    XT = init.resizeData(XT.detach().view(-1,d).cpu().numpy(),s,m)
    gO.writeVectorField(XT,vT,outpath+'target_to_source_velocityField' + str(res) + '.vtk')
    np.savez(outpath+'target_to_source_velocityField' + str(res) + '.npz',startPos=XT,endPos=vT,startW=qGridw.detach().cpu().numpy(),endW=wT)
    
    
    # forwards velocity field (from source to target)
    S = q0[numS:].detach().view(-1,d)
    maT = torch.max(S,axis=0).values
    miT = torch.min(S,axis=0).values
    x0T = torch.arange(miT[0] - 2.0*res, maT[0] + 2.0*res, res)
    x1T = torch.arange(miT[1] - 2.0*res, maT[1] + 2.0*res, res)
    x2T = torch.arange(miT[2] - 2.0*res, maT[2] + 2.0*res, res)
    X0T, X1T, X2T = torch.meshgrid((x0T,x1T,x2T),indexing='ij')
    XT = torch.stack((X0T.flatten(),X1T.flatten(),X2T.flatten()),axis=-1).type(dtype)
    numG = XT.shape[0]
    qGrid = XT.flatten()
    qGridw = torch.ones((numG)).type(dtype)    

    vT,wT = shootForwards(paramFile,variableFile,qGrid,qGridw)
    XT = init.resizeData(XT.detach().view(-1,d).cpu().numpy(),s,m)
    gO.writeVectorField(XT,vT,outpath+'source_to_target_velocityField' + str(res) + '.vtk')
    np.savez(outpath+'source_to_target_velocityField' + str(res) + '.npz',startPos=XT,endPos=vT,startW=qGridw.detach().cpu().numpy(),endW=wT)

    return

    
        