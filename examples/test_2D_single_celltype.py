import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../xmodmap')
sys_path.append('../xmodmap/io')
import initialize as init
import getInput as gI
import getOutput as gO


import torch

from sandbox.singleModalityHamiltonianATSCalibrated import *
from sandbox.analyzeOutput import *

np_dtype = "float32" # "float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor #DoubleTensor 
else:
    dtype = torch.FloatTensor


import nibabel as nib

def main():
    original = sys.stdout

    outpath='output/MERFISH_Celltypes/'

    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 
        
    S,nu_S = torch.load('../data/2D_single_celltype/source_2D_celltype.pt')
    T,nu_T = torch.load('../data/2D_single_celltype/target_2D_celltype.pt')
    sigmaRKHSlist,sigmaVarlist,gamma,d,labs,its,kScale,cA,cT,cS,dimEff,single = torch.load('../data/2D_single_celltype/parameters_2D_celltype.pt')
    N = S.shape[0]

    savedir = outpath + '2DdefaultParameters/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    
    sys.stdout = open(savedir+'test.txt','w')
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: ", sigmaRKHSlist)
    print("sigmaVar: ", sigmaVarlist)
    print("its: " + str(its))
    print("gamma: " + str(gamma))
    print("kScale: ", kScale)
    print("cA: ", cA)
    print("cT: ", cT)
    print("cS: ", cS)
    print("single: ", single)
    print("dimEff: ", dimEff)
    
    print("N " + str(N))
                
    Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,gamma,d,labs,savedir,its=its,kScale=kScale,cA=cA,cT=cT,cS=cS,dimEff=dimEff,single=single)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_S,axis=-1), np.argmax(nu_S,axis=-1)]
    imageValsT = [np.sum(nu_T,axis=-1), np.argmax(nu_T,axis=-1)]
    imageNamesS = ['weights', 'maxImageVal']
    imageNamesT = ['weights', 'maxImageVal']

    zeta_S = nu_S/(np.sum(nu_S,axis=-1)[...,None])
    zeta_T = nu_T/(np.sum(nu_T,axis=-1)[...,None])
    for i in range(labs):
        imageNamesT.append('zeta_' + str(i))
        imageValsT.append(zeta_T[:,i])
    for i in range(zeta_S.shape[-1]):
        imageValsS.append(zeta_S[:,i])
        imageNamesS.append('zeta_' + str(i))

    gO.writeVTK(S,imageValsS,imageNamesS,savedir+'testOutput_S.vtk',polyData=None)
    gO.writeVTK(T,imageValsT,imageNamesT,savedir+'testOutput_T.vtk',polyData=None)
    np.savez(savedir+'origST.npz',S=S,nu_S=nu_S,T=T,nu_T=T,zeta_S=zeta_S,zeta_T=zeta_T)
    pointList = np.zeros((S.shape[0]*len(Dlist),d))
    polyList = np.zeros((S.shape[0]*(len(Dlist)-1),3))
    polyList[:,0] = 2
    
    pointListG = np.zeros((Glist[0].shape[0]*len(Glist),d))
    polyListG = np.zeros((Glist[0].shape[0]*(len(Glist)-1),3))
    polyListG[:,0] = 2
    featList = np.zeros((S.shape[0]*len(Dlist),1))
    featListG = np.zeros((Glist[0].shape[0]*len(Glist),1))
    
    for t in range(len(Dlist)):
        D = Dlist[t]
        G = Glist[t]
        nu_D = nu_Dlist[t]
        nu_G = nu_Glist[t]
        zeta_D = nu_D/(np.sum(nu_D,axis=-1)[...,None])
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        for i in range(zeta_D.shape[-1]):
            imageValsD.append(zeta_D[:,i])
        gO.writeVTK(D,imageValsD,imageNamesS,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        gO.writeVTK(G,[nu_G],['Weights'],savedir+'testOutput_G' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez(savedir+'testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)
            pointList[int(t*len(D)):int((t+1)*len(D))] = D
            pointListG[int(t*len(G)):int((t+1)*len(G))] = G
            featList[int(t*len(D)):int((t+1)*len(D))] = np.squeeze(np.sum(nu_D,axis=-1))[...,None]
            featListG[int(t*len(G)):int((t+1)*len(G))] = np.squeeze(nu_G)[...,None]
        else:
            pointList[int(t*len(D)):int((t+1)*len(D))] = D
            pointListG[int(t*len(G)):int((t+1)*len(G))] = G
            featList[int(t*len(D)):int((t+1)*len(D))] = np.squeeze(np.sum(nu_D,axis=-1))[...,None]
            featListG[int(t*len(G)):int((t+1)*len(G))] = np.squeeze(nu_G)[...,None]
            polyList[int(t*len(D)):int((t+1)*len(D)),1] = np.arange(t*len(D),(t+1)*len(D))
            polyList[int(t*len(D)):int((t+1)*len(D)),2] = np.arange((t+1)*len(D),(t+2)*len(D))
            polyListG[int(t*len(G)):int((t+1)*len(G)),1] = np.arange(t*len(G),(t+1)*len(G))
            polyListG[int(t*len(G)):int((t+1)*len(G)),2] = np.arange((t+1)*len(G),(t+2)*len(G))


    jFile = gO.getJacobian(Dlist[-1],nu_S,nu_Dlist[-1],savedir+'testOutput_D10_jacobian.vtk')

    sigmaVar0 = sigmaVarlist[0].cpu().numpy()
    gO.writeVTK(pointList,[featList],['Weights'],savedir+'testOutput_curves.vtk',polyData=polyList)
    gO.writeVTK(pointListG,[featListG],['Weights'],savedir+'testOutput_grid.vtk',polyData=polyListG)
    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))
    getLocalDensity(S,nu_S,sigmaVar0,savedir+'density_S.vtk',coef=2.0)
    getLocalDensity(T,nu_T,sigmaVar0,savedir+'density_T.vtk',coef=2.0)
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar0,savedir+'density_D.vtk',coef=2.0)
    
    print("volumes of source, target, and deformed source")
    print(volS)
    print(volT)
    print(volD)
    print("total mass")
    wS = np.sum(nu_S)
    wT = np.sum(nu_T)
    wD = np.sum(nu_Dlist[-1])
    print(wS)
    print(wT)
    print(wD)
    print("total mass per total volume")
    print(wS/volS)
    print(wT/volT)
    print(wD/volD)
    
    optimalParams = np.load(savedir + 'testOutput_values.npz')
    A0 = optimalParams['A0']
    q0 = optimalParams['q0']
    tau0 = optimalParams['tau0']
    
    qx0 = np.reshape(q0[N:],(N,d))
    qw0 = np.reshape(q0[:N],(N,1))
    
    x = applyAandTau(qx0,qw0,A0,tau0)
    gO.writeVTK(x,[qw0],['weights'],savedir+'testOutput_D_ATau.vtk',polyData=None)
    getLocalDensity(x,nu_S,sigmaVar0,savedir+'density_D_ATau.vtk',coef=2.0)
    
    sys.stdout = original
    return

if __name__ == "__main__":
    main()