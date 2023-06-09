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

from singleModalityHamiltonianATSCalibrated import *
from analyzeOutput import *

np_dtype = "float32" # "float64"
dtype = torch.cuda.FloatTensor #DoubleTensor 

import nibabel as nib
###############################################################

def main():
    # User parameters
    sigmaRKHS = [0.05,0.01] # sigma of velocity field relative to support of 1
    sigmaVar = [0.05,0.01] # sigma of varifold norm relative to support of 1
    its = 50 # iterations of LBFGS steps (nb each step has 10 line searches)
    alphaSt = 't1_to_t2' # specification in output directory name
    
    # default parameters
    beta = None
    kScale=1.0
    extra=""
    cA=1.0
    cT=1.0
    cS=30.0
    d = 3 # dimensions of data (not effective dimensions)
    
    # Set these parameters according to relative decrease you expect in data attachment term
    # these should be based on approximately what the contribution compared to original cost is
    gamma = 1.0 #0.01 #10.0
    single=False
    
    original = sys.stdout

    # Path to output stdout and mapping results
    outpath='/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Synapses/'

    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 
        
    outpath = outpath + alphaSt + '/'
    
    if (not os.path.exists(outpath)):
        os.mkdir(outpath)
    
    # Load source and target (npz in form with files[0] as space coordinates and files[1] as features)
    sourceImage = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/Synapses/rsc01_reg_synapses_t1_formatted.npz'
    targetImage = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/Synapses/rsc01_reg_synapses_t2_formatted.npz'
    
    S,nu_S = gI.getFromFile(sourceImage)
    T,nu_T = gI.getFromFile(targetImage)
    
    S,T = init.checkZ(S,T)
    
    labs = nu_T.shape[-1]
    labS = nu_S.shape[-1]
    N = S.shape[0]

    savedir = outpath + '/output_dl_sig_its_albega_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(gamma) + str(beta) + '_' + str(N) + extra + '/'
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
    
    sigmaRKHSlist = []
    sigmaVarlist = []
    for sigg in sigmaRKHS:
        sigmaRKHSlist.append(torch.tensor(sigg).type(dtype))
    for sigg in sigmaVar:
        sigmaVarlist.append(torch.tensor(sigg).type(dtype))
        
    Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,torch.tensor(gamma).type(dtype),d,labs,savedir,its=its,kScale=torch.tensor(kScale).type(dtype),cA=torch.tensor(cA).type(dtype),cT=torch.tensor(cT).type(dtype),cS=cS,dimEff=3,single=single)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    # Default output plots sum of feature values and index of maximum feature value (1 based with 0 if weights = 0)
    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_S,axis=-1), (np.argmax(nu_S,axis=-1) + 1)*(np.sum(nu_S,axis=-1)>0)]
    imageValsT = [np.sum(nu_T,axis=-1), (np.argmax(nu_T,axis=-1) + 1)*(np.sum(nu_T,axis=-1)>0)]
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
    
    for t in range(len(Dlist)):
        D = Dlist[t]
        G = Glist[t]
        nu_D = nu_Dlist[t]
        nu_G = nu_Glist[t]
        zeta_D = nu_D/(np.sum(nu_D,axis=-1)[...,None])
        imageValsD = [np.sum(nu_D,axis=-1), (np.argmax(nu_D,axis=-1)+1)*(np.sum(nu_D,axis=-1)>0)]
        for i in range(zeta_D.shape[-1]):
            imageValsD.append(zeta_D[:,i])
        gO.writeVTK(D,imageValsD,imageNamesS,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        gO.writeVTK(G,[nu_G],['Weights'],savedir+'testOutput_G' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez(savedir+'testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)


    jFile = gO.getJacobian(Dlist[-1],nu_S,nu_Dlist[-1],savedir+'testOutput_D10_jacobian.vtk')

    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))
    getLocalDensity(S,nu_S,sigmaVar[0]*1000.0,savedir+'density_S.vtk',coef=2.0)
    getLocalDensity(T,nu_T,sigmaVar[0]*1000.0,savedir+'density_T.vtk',coef=2.0)
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar[0]*1000.0,savedir+'density_D.vtk',coef=2.0)
    
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
    getLocalDensity(x,nu_S,sigmaVar[0],savedir+'density_D_ATau.vtk',coef=0.25)
    
    sys.stdout = original
    return

if __name__ == "__main__":
    main()