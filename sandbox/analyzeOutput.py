import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

def getLocalDensity(Z,nu_Z,sigma,savename,coef=3):
    '''
    Compute local density in cube of size 2sigma x 2sigma x 2sigma
    '''
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
    for f in range(nu_Z.shape[-1]):
        imageNames.append('Feature' + str(f) + '_Density')
        imageDensity.append(cubes_mrna[:,f])
    vtf.writeVTK(centroidsPlot,imageDensity,imageNames,savename,polyData=None)
    return
   