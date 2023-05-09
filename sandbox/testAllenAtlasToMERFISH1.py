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

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch

from crossModalityHamiltonianATSCalibrated import *
from analyzeOutput import *

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float32" # "float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

import nibabel as nib

def main():
    d = 3
    labs = 2 # in target 
    labS = 3 # template
    sigmaRKHS = [0.05,0.2] # as of 3/16, should be fraction of total domain of S+T #[10.0]
    sigmaVar = [0.6,0.1,0.05,0.02] # as of 3/16, should be fraction of total domain of S+T #10.0
    its = 150
    alphaSt = 'MouseToMerfish'
    beta = None
    res=1.0
    kScale=1
    extra="sl484to212_alainParams"
    cA=1.0
    cT=1.0 # original is 0.5
    cS=1.0
    dimEff=2
    Csqpi = 10000.0
    
    # Set these parameters according to relative decrease you expect in data attachment term
    # these should be based on approximately what the contribution compared to original cost is
    gamma = 1.0 #10 for rescaling of varifold kernels by 0.6/sig #10.0
    
    original = sys.stdout

    outpath='/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/AllenMERFISH/' + alphaSt + '/'
    imgSource='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap.img'
    imgTarg='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/top20MI/_212_XnuX_lowToHighMI.npz.npz'
    imgTarg='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.05/_212_ZnuZ_lowToHighMI__optimalZnu_ZAllwC8.0_sig[0.05]_Nmax1500.0_Npart2000.0.npz'

    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 
    S,nu_S = gI.makeFromSingleChannelImage(imgSource,0.08,bg=[0],ds=8,axEx=[2,484],weights=torch.tensor(0.08**2).type(dtype))
    N = S.shape[0]
    labS = nu_S.shape[-1]
    
    T,nu_T = gI.getFromFile(imgTarg)
    
    labs = nu_T.shape[-1]
    labS = nu_S.shape[-1]
    cPi=torch.tensor(10.0/np.log(labs)).type(dtype) #0.1
    N = S.shape[0]

    # Trying Rotation manually 
    #Arot = init.get3DRotMatrix(torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0))
    #tauManual = torch.zeros((1,d)).type(dtype)
    #tauManual[0,0] = torch.tensor(2.0).type(dtype)
    #T,nu_T = init.applyAffine(T,nu_T,Arot,tauManual)
    
    #S,nu_S = init.scaleDataByVolumes(S,nu_S,T,nu_T,dRel=2) # dRel for what relative volume is 

    savedir = outpath + '/output_dl_sig_its_albega_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(gamma) + str(beta) + '_' + str(N) + str(cPi) + extra + '/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    
    sys.stdout = open(savedir+'test.txt','w')
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: " + str(sigmaRKHS))
    print("sigmaVar: " + str(sigmaVar))
    print("its: " + str(its))
    print("gammaA: " + str(gamma))
    print("beta: " + str(beta))
    
    print("N " + str(N))
    
    #Dlist, nu_Dlist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),d,labs,savedir,its=its,beta=beta)
    sigmaRKHSlist = []
    sigmaVarlist = []
    for sigg in sigmaRKHS:
        sigmaRKHSlist.append(torch.tensor(sigg).type(dtype))
    for sigg in sigmaVar:
        sigmaVarlist.append(torch.tensor(sigg).type(dtype))
        
    Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,torch.tensor(gamma).type(dtype),d,labs,savedir,its=its,kScale=torch.tensor(kScale).type(dtype),cA=torch.tensor(cA).type(dtype),cT=torch.tensor(cT).type(dtype),cS=cS,cPi=cPi,dimEff=dimEff,Csqpi=torch.tensor(Csqpi).type(dtype))
    
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

    vtf.writeVTK(S,imageValsS,imageNamesS,savedir+'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(T,imageValsT,imageNamesT,savedir+'testOutput_T.vtk',polyData=None)
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
        nu_Dpi = nu_DPilist[t]
        print("nu_D shape, ", nu_D.shape)
        print("nu_Dpi shape, ", nu_Dpi.shape)
        zeta_D = nu_D/(np.sum(nu_D,axis=-1)[...,None])
        zeta_Dpi = nu_Dpi / (np.sum(nu_Dpi,axis=-1)[...,None])
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        imageValsDPi = [np.sum(nu_Dpi,axis=-1),np.argmax(nu_Dpi,axis=-1)]
        for i in range(labs):
            imageValsDPi.append(zeta_Dpi[:,i])
        for i in range(zeta_D.shape[-1]):
            imageValsD.append(zeta_D[:,i])
        vtf.writeVTK(D,imageValsD,imageNamesS,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        vtf.writeVTK(D,imageValsDPi,imageNamesT,savedir+'testOutput_Dpi' + str(t) + '.vtk',polyData=None)
        vtf.writeVTK(G,[nu_G],['Weights'],savedir+'testOutput_G' + str(t) + '.vtk',polyData=None)
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



    vtf.writeVTK(pointList,[featList],['Weights'],savedir+'testOutput_curves.vtk',polyData=polyList)
    vtf.writeVTK(pointListG,[featListG],['Weights'],savedir+'testOutput_grid.vtk',polyData=polyListG)
    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))
    getLocalDensity(S,nu_S,sigmaVar[0],savedir+'density_S.vtk',coef=2.0)
    getLocalDensity(T,nu_T,sigmaVar[0],savedir+'density_T.vtk',coef=2.0)
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar[0],savedir+'density_D.vtk',coef=2.0)
    
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
    vtf.writeVTK(x,[qw0],['weights'],savedir+'testOutput_D_ATau.vtk',polyData=None)
    getLocalDensity(x,nu_S,sigmaVar[0],savedir+'density_D_ATau.vtk',coef=0.25)
    
    sys.stdout = original
    return

if __name__ == "__main__":
    main()