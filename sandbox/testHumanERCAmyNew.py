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

from fromScratchHamiltonianATCalibrated import *
from analyzeOutput import *
# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float64" # "float64"
dtype = torch.cuda.DoubleTensor #DoubleTensor 

import nibabel as nib

def main():
    d = 3
    labs = 3
    sigmaRKHS = [0.1] # as of 3/16, should be fraction of total domain of S+T #[10.0]
    sigmaVar = [0.2,0.05] # as of 3/16, should be fraction of total domain of S+T #10.0
    its = 15
    alphaSt = 'BEIALE-new'
    beta = None
    res=1.0
    kScale=1
    extra="cA1-csigma10"
    cA=1.0
    cT=1.0 # original is 0.5
    cS=10.0
    
    # Set these parameters according to relative decrease you expect in data attachment term
    # these should be based on approximately what the contribution compared to original cost is
    gammaA = 0.01 #10.0
    gammaT = 0.01 #0.1
    gammaU = 0.01*sigmaRKHS[0] # 1*2.0*sigmaRKHS[0]**3 sigmaRKHS
    
    original = sys.stdout

    outpath='/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Human/' + alphaSt + '/'
    imgPref='/cis/home/kstouff4/Documents/datasets/BIOCARD/SubsetFall2022/Segmentations/BEIALE/'

    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 

    imgFile = imgPref+'150428/AMYGDALA+ERC+TEC.img'
    im = nib.load(imgFile)
    imageO = np.asanyarray(im.dataobj).astype('float64')
    
    x0 = np.arange(imageO.shape[0])*res
    x1 = np.arange(imageO.shape[1])*res
    x2 = np.arange(imageO.shape[2])*res
    
    x0 -= np.mean(x0)
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    
    uniqueVals = np.unique(imageO)
    numUniqueMinus0 = len(uniqueVals)-1
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0).type(dtype),torch.tensor(x1).type(dtype),torch.tensor(x2).type(dtype),indexing='ij')
    nu_S = torch.zeros((X.shape[0],X.shape[1],X.shape[2],labs)).type(dtype)
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    listOfNu = []
    for u in range(1,len(uniqueVals)):
        nu_S[...,u-1] = torch.tensor((imageO == uniqueVals[u])).type(dtype)
        listOfNu.append(nu_S[...,u-1].flatten())
    nu_S = torch.stack(listOfNu,axis=-1).type(dtype)

    toKeep = nu_S.sum(axis=-1) > 0
    S = S[toKeep]
    nu_S = nu_S[toKeep]
    N = S.shape[0]
    
    # Debugging
    '''
    tauStart = torch.zeros((1,d)).type(dtype)
    tauStart[:,0] = -5.5
    tauStart[:,1] = 25.0
    tauStart[:,2] = -15.5
    T,nu_T = gi.applyAffine(S,nu_S,torch.eye(d).type(dtype),tauStart)
    '''
    
    
    imgFile = imgPref+'170516/AMYGDALA+ERC+TEC.img'
    im = nib.load(imgFile)
    imageO = np.asanyarray(im.dataobj).astype('float64')
    
    x0 = np.arange(imageO.shape[0])*res
    x1 = np.arange(imageO.shape[1])*res
    x2 = np.arange(imageO.shape[2])*res
    
    x0 -= np.mean(x0)
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    
    uniqueVals = np.unique(imageO)
    numUniqueMinus0 = len(uniqueVals)-1
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0).type(dtype),torch.tensor(x1).type(dtype),torch.tensor(x2).type(dtype),indexing='ij')
    nu_T = torch.zeros((X.shape[0],X.shape[1],X.shape[2],labs)).type(dtype)
    T = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    listOfNu = []
    for u in range(1,len(uniqueVals)):
        nu_T[...,u-1] = torch.tensor((imageO == uniqueVals[u])).type(dtype)
        listOfNu.append(nu_T[...,u-1].flatten())

    nu_T = torch.stack(listOfNu,axis=-1).type(dtype)
    toKeep = nu_T.sum(axis=-1) > 0
    T = T[toKeep]
    nu_T = nu_T[toKeep]
    
    # Trying Rotation manually 
    Arot = init.get3DRotMatrix(torch.tensor(np.pi/4.0),torch.tensor(np.pi/2.0),torch.tensor(3.0*np.pi/2.0))
    T,nu_T = init.applyAffine(T,nu_T,Arot,torch.zeros((1,d)).type(dtype))

    S, nu_S = gI.returnMultiplesSpace(S,nu_S,kScale)
    #S,nu_XS = ess.makeAllXandZ(imgPref+'150318/AMYGDALA+ERC+TEC.img', outpath+'ABEBER_150318_', thickness=-1, res=1.0,sig=0.1,C=-1,flip=False)
    #T,nu_XT = ess.makeAllXandZ(imgPref+'170831/AMYGDALA+ERC+TEC.img', outpath+'ABEBER_170831_', thickness=-1, res=1.0,sig=0.1,C=-1,flip=False)
    
    
    #print("min and max")
    #print(np.min(nu_XS))
    #print(np.max(nu_XS))
    
    #print("sizes are")
    #print(S.shape)
    #print(T.shape)
    
    #nu_S = torch.zeros((S.shape[0],3)).type(dtype)
    #nu_S[:,(nu_XS-1).astype(int)] = 1.0
    #nu_T = torch.zeros((T.shape[0],3)).type(dtype)
    #nu_T[:,(nu_XT-1).astype(int)] = 1.0
    
    #S = torch.tensor(S).type(dtype)
    #T = torch.tensor(T).type(dtype)
    '''
    npz = np.load('/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/output_dl_sig_its_albe_N-' + str(d) + str(labs) + '_' + str(100.0) + str(50.0) + '_' + str(its) + '_' + str(alpha) + str(beta) + '_' + str(1893) + '/' + 'testOutput.npz')
    S = torch.tensor(npz['D']).type(dtype)
    nu_S = torch.tensor(npz['nu_S']).type(dtype)
    T = torch.tensor(npz['T']).type(dtype)
    nu_T = torch.tensor(npz['nu_T']).type(dtype)
    '''
    
    N = S.shape[0]
    '''
    if (alpha is None):
        alpha = np.sqrt(N)
    if (gamma is None):
        gamma = np.sqrt(N)
    '''
    
    savedir = outpath + '/output_dl_sig_its_albega_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(gammaA) + str(beta) + str(gammaT) + str(gammaU) + '_' + str(N) + extra + '/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    
    sys.stdout = open(savedir+'test.txt','w')
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: " + str(sigmaRKHS))
    print("sigmaVar: " + str(sigmaVar))
    print("its: " + str(its))
    print("gammaA: " + str(gammaA))
    print("beta: " + str(beta))
    print("gamma: " + str(gammaT))
    print("gammaU: " + str(gammaU))
    
    print("N " + str(N))
    
    #Dlist, nu_Dlist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),d,labs,savedir,its=its,beta=beta)
    sigmaRKHSlist = []
    sigmaVarlist = []
    for sigg in sigmaRKHS:
        sigmaRKHSlist.append(torch.tensor(sigg).type(dtype))
    for sigg in sigmaVar:
        sigmaVarlist.append(torch.tensor(sigg).type(dtype))
        
    Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,torch.tensor(gammaA).type(dtype),d,labs,savedir,its=its,kScale=torch.tensor(kScale).type(dtype),cA=torch.tensor(cA).type(dtype),cT=torch.tensor(cT).type(dtype),cS=cS)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_S,axis=-1), np.argmax(nu_S,axis=-1)]
    imageValsT = [np.sum(nu_T,axis=-1), np.argmax(nu_T,axis=-1)]

    zeta_S = nu_S/(np.sum(nu_S,axis=-1)[...,None])
    zeta_T = nu_T/(np.sum(nu_T,axis=-1)[...,None])
    for i in range(labs):
        imageNames.append('zeta_' + str(i))
        imageValsS.append(zeta_S[:,i])
        imageValsT.append(zeta_T[:,i])

    vtf.writeVTK(S,imageValsS,imageNames,savedir+'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(T,imageValsT,imageNames,savedir+'testOutput_T.vtk',polyData=None)
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
        for i in range(labs):
            imageValsD.append(zeta_D[:,i])
        vtf.writeVTK(D,imageValsD,imageNames,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
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