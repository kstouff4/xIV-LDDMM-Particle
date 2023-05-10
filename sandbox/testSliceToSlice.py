import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
import os
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init
import getInput as gI

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
import sliceToSliceAlignment as sts

np_dtype = "float32" #"float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 


def main():
    sigma = 0.5 # in mm 
    its = 50
    
    wholeX = np.load('/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_[28-56-111-101-47]_mmRedone_XoneHot.npz')
    zs = np.unique(wholeX[wholeX.files[0]][...,-1])
    
    filesO = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/Redo__optimalZnu_ZAllwC8.0_sig[0.2]_Nmax1500.0_Npart2000.0.npz']
    
    extra = "BarSeq/"
    
    #wholeX = np.load('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ_sig0.05.npz')
    #zs = np.unique(wholeX[wholeX.files[0]][...,-1])
    #filesO = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ__optimalZnu_ZAllwC1.2_sig[0.1]_Nmax1500.0_Npart2000.0.npz']
    
    #extra = "AllenMerfish/"
                       
    original = sys.stdout
    
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir) 
    
    savedir = savedir + extra
    
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
        
    savedir = savedir + str(sigma) + '/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
        
    sys.stdout = open(savedir+'test.txt','w')
    Slist, nu_Slist, slIndex = sts.getSlicesFromWhole(filesO[0],zs)
    print(len(Slist))
    print(len(nu_Slist))
    
    Snew, thetas, taus = sts.align(Slist,nu_Slist,torch.tensor(sigma).type(dtype),its,savedir,norm=False)
    totalPoints = slIndex.shape[0]
    SnewTotal = np.zeros((totalPoints,3))
    nuSTotal = np.zeros((totalPoints,nu_Slist[0].shape[-1]))
    count = 0
    for s in range(len(Snew)):
        S = Snew[s].detach().cpu().numpy()
        nu_S = nu_Slist[s].detach().cpu().numpy()
        SnewTotal[count:count+S.shape[0],0:2] = S
        SnewTotal[count:count+S.shape[0],2] = zs[s]
        nuSTotal[count:count+S.shape[0],...] = nu_S
        np.savez(savedir+'slice_' + str(s) + '.npz',S=SnewTotal[count:count+S.shape[0]],nu_S=nu_S)
        vtf.writeVTK(SnewTotal[count:count+S.shape[0]],[np.sum(nu_S,axis=-1),np.argmax(nu_S,axis=-1)],['totalMass','maxVal'],savedir+'slice_' + str(s) + '.vtk',polyData=None)
        count = count + S.shape[0]
    np.savez(savedir+'allSlices.npz',S=SnewTotal,nu_S=nuSTotal)
    imageNames = ['totalMass','maxVal']
    imageVals = [np.sum(nuSTotal,axis=-1),np.argmax(nuSTotal,axis=-1)]
    zetaTotal = nuSTotal/np.sum(nuSTotal,axis=-1)[...,None]
    for f in range(nuSTotal.shape[-1]):
        imageNames.append('zeta_' + str(f))
        imageVals.append(zetaTotal[:,f])
    vtf.writeVTK(SnewTotal,imageVals,imageNames,savedir+'allSlices.vtk',polyData=None)

    sys.stdout = original
    return

if __name__ == "__main__":
    main()