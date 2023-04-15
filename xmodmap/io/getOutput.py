import numpy as np
from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
dtype = torch.cuda.FloatTensor # Double Tensor 

def computeRegionStatisticsImage(npzFile,labels):
    '''
    npz file should have nu_Scomb = original source compartments (should be 1 label per particle)
    nu_D = deformed source 
    labels = list of named labels per nu_Scomb
    '''
    info = np.load(npzFile)
    nu_D = info['nu_D']
    nu_Scomb = info['nu_Scomb']
    w_Scomb = np.sum(nu_Scomb,axis=-1)
    startW = np.sum(nu_Scomb,axis=0)
    wD = np.sum(nu_D,axis=-1)
    zeta_Scomb = nu_Scomb/w_Scomb[...,None]
    nu_Dcomb = zeta_Scomb*wD
    D = info['D']
    nu_DcombRat = zeta_Scomb*(wD/w_Scomb)[...,None]
    
    imageNames = []
    imageVals = []
    imageNames.append('maxVal')
    imageVals.append(np.argmax(nu_Dcomb,axis=-1))
    imageNames.append('mass')
    imageVals.append(wD)
    imageNames.append('changeInMass')
    imageVals.append(wD/w_Scomb)
    for l in range(len(labels)):
        imageNames.append(labels[l])
        imageVals.append(nu_Dcomb[:,l])
    
    vtf.writeVTK(D,imageVals,imageNames,npzFile.replace('.npz','_regions.vtk'),polyData=None)
    f,ax = plt.subplots(2,1)
    yErrMass = np.std(nu_Dcomb,axis=0)
    yMeanMass = np.mean(nu_Dcomb,axis=0)
    yErrRatio = np.std(nu_DcombRat,axis=0)
    yMeanRatio = np.mean(nu_DcombRat,axis=0)
    
    ax[0].bar(np.arange(len(yErrMass)),yMeanMass,yErr=yErrMass)
    ax[0].set_ylabel('Total Mass (mm$^3$)')
    ax[0].set_xticklabels(labels)
    ax[1].bar(np.arange(len(yErrRatio)),yMeanRatio,yErr=yErrMeanRatio)
    ax[1].set_ylabel('Change in Mass')
    ax[1].set_xticklabels(labels)
    f.savefig(npzFile.replace('.npz','_regionsStats.png'),dpi=300)
    return
    

