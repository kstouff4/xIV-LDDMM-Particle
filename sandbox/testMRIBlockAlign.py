import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init
import getInput as gI

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float32" # "float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

import nibabel as nib
import alignMRIBlocks as amb

##########################################################################

def main():
    sigmaVar = 1.0
    gammaR = 0.01
    gammaC = 1.0
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Human/Exvivohuman_11T/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    savedir = savedir + 'JOG57/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
        
    zCoordsO = [torch.tensor(149.0).type(dtype),torch.tensor(0.0).type(dtype),torch.tensor(-140.0).type(dtype)]
    
    b1S,b1nu_S = gI.getFromFile('/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57/JOG57_1_ap.npz')
    b2S,b2nu_S = gI.getFromFile('/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57/JOG57_2_ap.npz')
    b3S,b3nu_S = gI.getFromFile('/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57/JOG57_3_ap.npz')
    b4S,b2nu_S = gI.getFromFile('/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57/JOG57_4_ap.npz')
    
    ASlistScalenp,Asnp,tsnp,s,m = amb.callOptimize([b1S,b2S,b3S,b4S],[b1nu_S,b2nu_S,b3nu_S,b4nu_S],sigmaVar,gammaR,gammaC,savedir, zCoordsO,its=100,d=3,numVars=6)
    
    imagenames = ['TotalMass','MaxBin','Bin_0','Bin_1','Bin_2']
    
    for i in range(len(Asnp)):
        imageVals = []
        imageVals.append(np.sum(Asnp[i],axis=-1))
        imageVals.append(np.argmax(Asnp[i],axis=-1))
        imageVals.append(Asnp[i][:,0])
        imageVals.append(Asnp[i][:,1])
        imageVals.append(Asnp[i][:,2])
        vtf.writeVTK(ASlistScalenp[i],imageVals,imagenames,savedir + 'transformSeg_' + str(i) + '.vtk',polyData=None)
    return

if __name__ == "__main__":
    main()


