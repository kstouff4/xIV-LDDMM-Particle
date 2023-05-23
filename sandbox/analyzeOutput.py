import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import scipy as sp
from scipy import linalg

import nibabel as nib

def getLocalDensity(Z,nu_Z,sigma,savename,coef=3):
    '''
    Compute local density in cube of size 2sigma x 2sigma x 2sigma
    '''
    if (len(nu_Z.shape) < 2):
        nu_Z = nu_Z[...,None]
    if (nu_Z.shape[-1] == 1):
        nu_Z = np.zeros_like(nu_Z) + 1.0
        print("nu_Z shape is, ", nu_Z.shape)
    cSize = coef*sigma
    bbSize = np.round(1+np.max(Z,axis=(0,1)) - np.min(Z,axis=(0,1))-1)
    
    coords_labels = np.floor((Z - np.floor(np.min(Z,axis=0)))/cSize).astype(int) # minimum number of cubes in x and y 
    totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)*(np.max(coords_labels[:,2])+1)
    xC = np.arange(np.max(coords_labels[:,0])+1)*cSize + np.floor(np.min(Z[:,0])) + cSize/2.0
    yC = np.arange(np.max(coords_labels[:,1])+1)*cSize + np.floor(np.min(Z[:,1])) + cSize/2.0
    zC = np.arange(np.max(coords_labels[:,2])+1)*cSize + np.floor(np.min(Z[:,2])) + cSize/2.0
    
    XC,YC, ZC = np.meshgrid(xC,yC,zC,indexing='ij')
    cubes_centroids = np.stack((XC,YC,ZC),axis=-1)
    cubes_indices = np.reshape(np.arange(totCubes),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    coords_labels_tot = cubes_indices[coords_labels[:,0],coords_labels[:,1],coords_labels[:,2]]
    
    cubes_mrna = np.zeros((totCubes,nu_Z.shape[-1]))
    for c in range(totCubes):
        cubes_mrna[c,:] = np.sum(nu_Z[coords_labels_tot == c,:],axis=0)
    centroidsPlot = np.zeros((totCubes,3))
    centroidsPlot[:,0] = np.ravel(XC)
    centroidsPlot[:,1] = np.ravel(YC)
    centroidsPlot[:,2] = np.ravel(ZC)
    
    cubes_mrna = cubes_mrna/(cSize**3)
    imageNames = []
    imageDensity = []
    imageNames.append('TotalDensity')
    imageDensity.append(np.sum(cubes_mrna,axis=-1))
    if (nu_Z.shape[-1] > 1):
        for f in range(nu_Z.shape[-1]):
            imageNames.append('Feature' + str(f) + '_Density')
            imageDensity.append(cubes_mrna[:,f])
    vtf.writeVTK(centroidsPlot,imageDensity,imageNames,savename,polyData=None)
    return

def getCompareDensity(T,nu_T,D,nu_Dpi,sigma,savedir,coef=3):
    '''
    Compute local density in cube of size 2sigma x 2sigma x 2sigma; save as segmentation image (maxVal) and intensity (weights)
    '''
    cSize = coef*sigma
    miT = np.min(T,axis=0)
    maT = np.max(T,axis=0)
    miD = np.min(D,axis=0)
    maD = np.max(D,axis=0)
    
    mi = np.vstack((miT,miD))
    ma = np.vstack((maT,maD))
    mi = np.min(mi,axis=0)
    ma = np.max(ma,axis=0)
    bbSize = np.round((1+ma) - (mi-1))
    
    coords_labels = np.floor((T - np.floor(np.min(T,axis=0)))/cSize).astype(int) # minimum number of cubes in x and y 
    coords_labels = np.ceil((np.ceil(ma) - np.floor(mi))/cSize).astype(int) # number of cubes in x, y, and z
    totCubes = (np.max(coords_labels[:,0]))*(np.max(coords_labels[:,1]))*(np.max(coords_labels[:,2]))
    xC = np.arange(np.max(coords_labels[:,0]))*cSize + np.floor(mi[0]) + cSize/2.0
    yC = np.arange(np.max(coords_labels[:,1]))*cSize + np.floor(mi[1]) + cSize/2.0
    zC = np.arange(np.max(coords_labels[:,2]))*cSize + np.floor(mi[2]) + cSize/2.0 # center in middle of cube 
    
    XC,YC, ZC = np.meshgrid(xC,yC,zC,indexing='ij') # physical coordinates 
    cubes_centroids = np.stack((XC,YC,ZC),axis=-1)
    cubes_indices = np.reshape(np.arange(totCubes),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    coords_labels_tot = cubes_indices[coords_labels[:,0],coords_labels[:,1],coords_labels[:,2]]
    
    # assign each measure to 1 cube
    coords_labelsT = np.floor((T - np.floor(mi))/cSize).astype(int)
    coords_labelsD = np.floor((D - np.floor(mi))/cSize).astype(int)
    
    cubes_nuT = np.zeros((totCubes,nu_T.shape[-1]))
    cubes_nuD = np.zeros((totCubes,nu_D.shape[-1]))
    
    for c in range(totCubes):
        cubes_nuT[c,:] = np.sum(nu_T[coords_labelsT == c,:],axis=0)
        cubes_nuD[c,:] = np.sum(nu_D[coords_labelsD == c,:],axis=0)
        
    # save densities as images 
    densT = np.reshape(np.sum(cubes_nuT,axis=-1),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    densD = np.reshape(np.sum(cubes_nuD,axis=-1),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    empty_header = nib.Nifti1Header()
    densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
    densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
    densTDim = nib.Nifti1Image(np.sqrt((densT-densD)**2),np.eye(4),empty_header)
    
    nib.save(densTim,savedir + 'Tdensity.nii.gz')
    nib.save(densDim,savedir + 'Ddensity.nii.gz')
    nib.save(densTDim,savedir + 'TDdiffdensity.nii.gz')
    
    densT = np.reshape(np.argmax(cubes_nuT,axis=-1),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    densD = np.reshape(np.argmax(cubes_nuD,axis=-1),(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
    empty_header = nib.Nifti1Header()
    densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
    densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
    densTDim = nib.Nifti1Image((densT - densD != 0).astype(int),np.eye(4),empty_header)
    
    nib.save(densTim,savedir + 'Tmaxval.nii.gz')
    nib.save(densDim,savedir + 'Dmaxval.nii.gz')
    nib.save(densTDim,savedir + 'TDdiffmaxval.nii.gz')
    
    for i in range(nu_T.shape[-1]):
        densT = np.reshape(cubes_nuT[...,i],(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
        densD = np.reshape(cubes_nuD[...,i],(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_centroids.shape[2]))
        empty_header = nib.Nifti1Header()
        densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
        densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
        densTDim = nib.Nifti1Image((densT - densD != 0).astype(int),np.eye(4),empty_header)
    
        nib.save(densTim,savedir + 'Tnu_' + str(i) + '.nii.gz')
        nib.save(densDim,savedir + 'Dnu_' + str(i) + '.nii.gz')
        nib.save(densTDim,savedir + 'TDdiffnu_' + str(i) + '.nii.gz')
    
    np.savez(savedir + 'TD_values.npz',cubes_nuT=cubes_nuT,cubes_nuD=cubes_nuD,cubes_centroids=cubes_centroids,XC=XC,YC=YC,ZC=ZC,cSize=cSize)

    '''
    centroidsPlot = np.zeros((totCubes,3))
    centroidsPlot[:,0] = np.ravel(XC)
    centroidsPlot[:,1] = np.ravel(YC)
    centroidsPlot[:,2] = np.ravel(ZC)
    
    cubes_mrna = cubes_mrna/(cSize**3)
    imageNames = []
    imageDensity = []
    imageNames.append('TotalDensity')
    imageDensity.append(np.sum(cubes_mrna,axis=-1))
    if (nu_Z.shape[-1] > 1):
        for f in range(nu_Z.shape[-1]):
            imageNames.append('Feature' + str(f) + '_Density')
            imageDensity.append(cubes_mrna[:,f])
    vtf.writeVTK(centroidsPlot,imageDensity,imageNames,savename,polyData=None)
    '''
    return

def applyAandTau(q_x,q_w,A,tau):
    '''
    q_x indicates the original positions of the source and q_w the original weights 
    arguments are numpy arrays 
    '''
    x_c0 = np.sum(q_w*q_x,axis=0)/np.sum(q_w)
    x = (q_x-x_c0)@((sp.linalg.expm(A)).T) + tau
    
    return x
   