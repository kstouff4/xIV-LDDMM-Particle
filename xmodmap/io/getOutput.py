import numpy as np
from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
dtype = torch.cuda.FloatTensor # Double Tensor 

from matplotlib import pyplot as plt

def computeRegionStatisticsImage(npzFile,labels,plotOriginal=False):
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
    nu_Dcomb = zeta_Scomb*wD[...,None]
    D = info['D']
    nu_DcombRat = zeta_Scomb*(wD/w_Scomb)[...,None]
    
    imageNames = []
    imageVals = []
    imageNames.append('maxVal')
    imageVals.append(np.argmax(nu_Dcomb,axis=-1))
    imageNames.append('mass')
    imageVals.append(wD)
    imageNames.append('RatioInMass')
    imageVals.append(wD/w_Scomb)
    imageNames.append('PercentChangeInMass')
    imageVals.append(100.0*(wD - w_Scomb)/w_Scomb)
    for l in range(len(labels)):
        imageNames.append(labels[l])
        imageVals.append(nu_Dcomb[:,l])
    
    vtf.writeVTK(D,imageVals,imageNames,npzFile.replace('.npz','_regions.vtk'),polyData=None)
    vtf.writeVTK(info['S'],imageVals,imageNames,npzFile.replace('.npz','_regionsInOrigCoords.vtk'),polyData=None)
    f,ax = plt.subplots(3,1,figsize=(6,10))
    yErrMass = np.std(nu_Dcomb,axis=0)
    yMeanMass = np.sum(nu_Dcomb,axis=0)
    yErrRatio = np.std(nu_DcombRat,axis=0)
    yMeanRatio = np.sum(nu_DcombRat,axis=0)
    yPerChange = 100.0*(yMeanMass - startW)/startW
    
    labelsNew = []
    labelsNew.append('')
    for l in labels:
        labelsNew.append(l)
    ax[0].bar(np.arange(len(yErrMass)),yMeanMass)
    ax[0].set_ylabel('Total Mass (mm$^3$)')
    ax[0].set_xticklabels(labelsNew)
    ax[1].bar(np.arange(len(yErrRatio)),yMeanRatio)
    ax[1].set_ylabel('Ratio in Mass')
    ax[1].set_xticklabels(labelsNew)
    ax[2].bar(np.arange(len(yErrRatio)),yPerChange)
    ax[2].set_ylabel('Percent Change in Mass')
    ax[2].set_xticklabels(labelsNew)
    f.savefig(npzFile.replace('.npz','_regionsStats.png'),dpi=300)
    
    if plotOriginal:
        vtf.writeVTK(info['S'],[np.argmax(nu_Scomb,axis=-1),np.sum(nu_Scomb,axis=-1)],['MaxVal','TotalMass'],npzFile.replace('.npz','_origNu_Scomb.vtk'),polyData=None)
    return info['S'], nu_Dcomb

def analyzeLongitudinalMass(listOfNu,S,ages,savename,labels):
    '''
    Assume list is ordered in time
    Assume each particle is 100% one type
    '''
    X = np.ones((len(ages),2))
    X[:,1] = ages
    
    coef = np.linalg.inv(X.T@X)@X.T #
    Y = np.zeros((len(ages),listOfNu[0].shape[0])) # number of particles
    for l in range(len(ages)):
        Y[l,:] = np.sum(listOfNu[l],axis=-1)[None,...] # get total mass
    beta = coef@Y
    imageNames = []
    imageVals = []
    imageNames.append("SlopeEst_ChangeInMassPerYear")
    imageVals.append(beta[1,:].T)
    imageNames.append("SlopeEst_PercChangeInMassPerYear_OfStart")
    imageVals.append(100.0*beta[1,:].T/Y[0,:].T)
    imageNames.append("Perc_ChangeInMassPerYear")
    imageVals.append((100.0*(Y[-1,:]-Y[0,:])/((Y[0,:])*(ages[-1]-ages[0]))).T)
    imageNames.append("ChangeInMassPerYear")
    imageVals.append(((Y[-1,:]-Y[0,:])/((ages[-1]-ages[0]))).T)
    vtf.writeVTK(S,imageVals,imageNames,savename,polyData=None)
    
    f,ax = plt.subplots(len(labels),1,figsize=(6,12))
    slopes = beta[1,:]
    slopes = []
    for l in range(len(labels)):
        slopes.append(beta[1,np.squeeze(listOfNu[0][:,l] > 0)].T)
    for i in range(len(labels)):
        ax[i].hist(slopes[i],label="Mean $\pm$ Std = {0:.6f} $\pm$ {1:.6f}".format(np.mean(slopes[i]),np.std(slopes[i])))
        ax[i].set_xlabel('Estimated (LS) Change In Mass Per Year')
        ax[i].set_ylabel(labels[i] + ' Frequency')
        ax[i].set_xlim([-0.025,0.025])
        ax[i].legend()
    f.savefig(savename.replace('.vtk','_stats.png'),dpi=300)
    np.savez(savename.replace('.vtk','.npz'),beta=beta,coef=coef,Y=Y)
    return

def getJacobian(D,nu_S,nu_D,savename):
    j = np.sum(nu_D,axis=-1)/np.sum(nu_S,axis=-1)
    imageNames = ['maxVal','totalMass','jacobian']
    imageVals = [np.argmax(nu_D,axis=-1), np.sum(nu_D,axis=-1), j]
    vtf.writeVTK(S,imageVals,imageNames,savename,polyData=None)
    return

def splitZs(T,nu_T,D,nu_D,savename,units=10):
    # split target, and deformed source along Z axis (last one) into units
    ma = np.max(T,axis=-1)[-1]
    mi = np.min(T,axis=-1)[-1]
    mas = np.max(D,axis=-1)[-1]
    mis = np.min(D,axis=-1)[-1]
    mi = min(mi,mis)
    ma = max(ma,mas)
    
    interval = np.round(ma-mi/units)
    imageNamesT = ['maxVal', 'totalMass']
    imageNamesD = ['maxVal','totalMass']
    for f in range(nu_T.shape[-1]):
        imageNamesT.apppend('zeta_' + str(f))
    for f in range(nu_D.shape[-1]):
        imageNamesD.append('zeta_' + str(f))
    
    for i in range(interval):
        iT = (T[...,-1] >= mi + i*interval)*(T[...,-1] < mi + (i+1)*interval)
        nu_Ts = nu_T[iT,...]
        imageVals = [np.argmax(nu_Ts,axis=-1),np.sum(nu_Ts,axis=-1)]
        for f in range(nu_T.shape[-1]):
            imageVals.append(nu_Ts[:,f])
        vtf.writeVTK(T[iT,...],imageVals,imageNames,savename+'_zunit' + str(i) + '.vtk',polyData=None)
        iD = (D[...,-1] >= mi + i*interval)*(D[...,-1] < mi + (i+1)*interval)
        nu_Ds = nu_D[iD,...]
        imageVals = [np.argmax(nu_Ds,axis=-1),np.sum(nu_Ds,axis=-1)]
        for f in range(nu_D.shape[-1]):
            imageVals.append(nu_Ds[:,f])
        vtf.writeVTK(D[iD,...],imageVals,imageNames,savename+'_zunit'+str(i) + '.vtk',polyData=None)
    return
                   
                        
    
        
    

