import os
import sys
from sys import path as sys_path
sys_path.append('..')

import xmodmap.io.getOutput as gO

import torch

from sandbox.crossModalityHamiltonianATSCalibrated_Boundary import *
from sandbox.analyzeOutput import *

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float32" # "float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor #DoubleTensor
    map_location = lambda storage, loc: storage.cuda(0)
else:
    dtype = torch.FloatTensor
    map_location = torch.device('cpu')

torch.set_default_tensor_type(dtype)

def main():
    original = sys.stdout

    savedir = os.path.join('output', 'RatToMouse', '2DdefaultParameters')
    datadir = os.path.join('..', 'data', '2D_cross_ratToMouseAtlas')

    os.makedirs(savedir, exist_ok=True)


    S, nu_S = torch.load(os.path.join(datadir, 'source_2D_ratAtlas.pt'), map_location=map_location)
    T, nu_T = torch.load(os.path.join(datadir, 'target_2D_mouseAtlas.pt'), map_location=map_location)
    sigmaRKHSlist, sigmaVarlist, gamma, d, labs, its, kScale, cA, cT, cS, cPi, dimEff, Csqpi, Csqlamb, eta0, lamb0, single = torch.load(
        os.path.join(datadir, 'parameters_2D_ratToMouseAtlas.pt'), map_location=map_location)


    N = S.shape[0]


    sys.stdout = open(os.path.join(savedir, 'test.log'),'w')
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
    print("cPi: ", cPi)
    print("dimEff: ", dimEff)
    print("Csqpi: ", Csqpi)
    print("Csqlamb: ", Csqlamb)
    print("eta0: ", eta0)
    print("lamb0: ", lamb0)
    print("single: ", single)
    print("N " + str(N))
    
    Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist, Tlist, nu_Tlist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,gamma,d,labs,savedir,its=its,kScale=kScale,cA=cA,cT=cT,cS=cS,cPi=cPi,dimEff=dimEff,Csqpi=Csqpi,Csqlamb=Csqlamb,eta0=eta0,lambInit=lamb0,loadPrevious=None,single=single)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()
    nu_Td = nu_Tlist[-1]
    Td = Tlist[-1]

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_S,axis=-1), np.argmax(nu_S,axis=-1)]
    imageValsT = [np.sum(nu_T,axis=-1), np.argmax(nu_T,axis=-1)]
    imageNamesS = ['weights', 'maxImageVal']
    imageNamesT = ['weights', 'maxImageVal']

    zeta_S = nu_S/(np.sum(nu_S,axis=-1)[...,None])
    zeta_T = nu_T/(np.sum(nu_T,axis=-1)[...,None])
    w_S = np.sum(nu_S,axis=-1)[...,None]
    w_T = np.sum(nu_T,axis=-1)[...,None]
    zeta_S[np.squeeze(w_S == 0),...] = 0
    zeta_T[np.squeeze(w_T == 0),...] = 0
    
    for i in range(labs):
        imageNamesT.append('zeta_' + str(i))
        imageValsT.append(zeta_T[:,i])
    for i in range(zeta_S.shape[-1]):
        imageValsS.append(zeta_S[:,i])
        imageNamesS.append('zeta_' + str(i))

    gO.writeVTK(S,imageValsS,imageNamesS,savedir+'testOutput_S.vtk',polyData=None)
    gO.writeVTK(T,imageValsT,imageNamesT,savedir+'testOutput_T.vtk',polyData=None)
    np.savez(savedir+'origST.npz',S=S,nu_S=nu_S,T=T,nu_T=nu_T,zeta_S=zeta_S,zeta_T=zeta_T)
    imageValsT[0] = np.sum(nu_Td,axis=-1)
    gO.writeVTK(Td,imageValsT,imageNamesT,savedir+'testOutput_Td.vtk',polyData=None)

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
        zeta_D = nu_D/(np.sum(nu_D,axis=-1)[...,None])
        zeta_Dpi = nu_Dpi / (np.sum(nu_Dpi,axis=-1)[...,None])
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        imageValsDPi = [np.sum(nu_Dpi,axis=-1),np.argmax(nu_Dpi,axis=-1)]
        for i in range(labs):
            imageValsDPi.append(zeta_Dpi[:,i])
        for i in range(zeta_D.shape[-1]):
            imageValsD.append(zeta_D[:,i])
        gO.writeVTK(D,imageValsD,imageNamesS,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        gO.writeVTK(D,imageValsDPi,imageNamesT,savedir+'testOutput_Dpi' + str(t) + '.vtk',polyData=None)
        gO.writeVTK(G,[nu_G],['Weights'],savedir+'testOutput_G' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez(savedir+'testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D,nu_Dpi=nu_Dpi)
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


    sigmaVar0 = sigmaVarlist[0].cpu().numpy()
    gO.writeVTK(pointList,[featList],['Weights'],savedir+'testOutput_curves.vtk',polyData=polyList)
    gO.writeVTK(pointListG,[featListG],['Weights'],savedir+'testOutput_grid.vtk',polyData=polyListG)
    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))
    getLocalDensity(S,nu_S,sigmaVar0,savedir+'density_S.vtk',coef=2.0)
    getLocalDensity(T,nu_T,sigmaVar0,savedir+'density_T.vtk',coef=2.0)
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar0,savedir+'density_D.vtk',coef=2.0)
    
    jFile = gO.getJacobian(Dlist[-1],nu_S,nu_Dlist[-1],savedir+'testOutput_D10_jacobian.vtk')
    gO.splitZs(T,nu_T,Dlist[-1],nu_DPilist[-1],savedir+'testOutput_Dpi10',units=15,jac=jFile)
    
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