import numpy as np
import torch
from pykeops.torch import Vi,Vj
np_dtype = "float32" #"float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor #DoubleTensor 
else:
    dtype = torch.FloatTensor

import nibabel as nib

import sys
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../xmodmap/')
sys_path.append('../sandbox')
sys_path.append('../xmodmap/io/')
import initialize as init
import getOutput as gO

from sandbox.crossModalityHamiltonianATSCalibrated_Boundary import GaussKernelHamiltonian, ShootingGrid, ShootingBackwards
from sandbox.saveState import loadParams, loadVariables

def main():
    NT = 100
    dirTest = '../sandbox/HumanCrossModality/B2toB5_NoBoundary/output_dl_sig_its_csgamma_N-36_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_100_10.0100.01.0_1670/'
    paramFile = dirTest + 'State__params.pt'
    variableFile = dirTest + 'State__variables.pt'
    
    uCoeff,sigmaRKHS,sigmaVar,beta,d,labs,numS,pTilde,gamma,cA,cT,cPi,dimE,single = loadParams(paramFile)
    q0,p0,Ttilde,w_T,s,m = loadVariables(variableFile)
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS,d=d,uCoeff=uCoeff)
    
    x = torch.arange(5).type(dtype)
    x = x - torch.mean(x)
    XG,YG,ZG = torch.meshgrid((x,x,x),indexing='ij')
    qGrid = torch.stack((XG.flatten(),YG.flatten(),ZG.flatten()),axis=-1).type(dtype)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG)).type(dtype) 

    listpq = ShootingGrid(p0,q0,qGrid,qGridw,Kg,sigmaRKHS,d,numS,uCoeff,cA,cT,dimE,single=single,nt=NT)
    listBack = ShootingBackwards(listpq[-1][0],listpq[-1][1],listpq[-1][2],listpq[-1][3],Kg,sigmaRKHS,d,numS,uCoeff,cA,cT,dimE,single,nt=NT)
        
    q0n = init.resizeData(q0[numS:].detach().view(-1,d).cpu().numpy(),s,m)
    q0n_dup = init.resizeData(listBack[-1][1][numS:].detach().view(-1,d).cpu().numpy(),s,m)
    w0n = q0[:numS].detach().view(-1,1).cpu().numpy()
    w0n_dup = listBack[-1][1][:numS].detach().view(-1,1).cpu().numpy()
    
    gO.writeVTK(q0n,[w0n],['weights'],dirTest+'test_originalq0_' + str(NT) + '.vtk',polyData=None)
    gO.writeVTK(q0n_dup,[w0n_dup],['weights'],dirTest+'test_backwardsShotq0_' + str(NT) + '.vtk',polyData=None)
        
    qgn = qGrid.detach().view(-1,d).cpu().numpy()
    qgn_dup = listBack[-1][2].detach().view(-1,d).cpu().numpy()
    wgn = qGridw.detach().view(-1,1).cpu().numpy()
    wgn_dup = listBack[-1][3].detach().view(-1,1).cpu().numpy()
    
    gO.writeVTK(qgn,[wgn],['weights'],dirTest+'test_originalqGrid_' + str(NT) + '.vtk',polyData=None)
    gO.writeVTK(qgn_dup,[wgn_dup],['weights'],dirTest+'test_backwardsShotqGrid_' + str(NT) + '.vtk',polyData=None)
    
    print("sum of square distances between state: ", np.sum((q0n-q0n_dup)**2))
    print("avg of square distances between state: ", np.sum((q0n-q0n_dup)**2)/q0n.shape[0])
    print("sum of square distances between grid: ", np.sum((qgn-qgn_dup)**2))
    print("avg of square distances between grid: ", np.sum((qgn-qgn_dup)**2)/qgn.shape[0])
    print("sum of square distances between weights: ", np.sum((w0n-w0n_dup)**2))
    print("sum of square distances between weights: ", np.sum((wgn-wgn_dup)**2))

    return

if __name__ == "__main__":
    main()
    
    
