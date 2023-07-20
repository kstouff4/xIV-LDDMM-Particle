import sys
from sys import path as sys_path
sys_path.append('..')

import torch

legacy = False

if legacy:
    from sandbox.crossModalityHamiltonianATSCalibrated_Boundary_legacy import *
else:
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

    savedir = os.path.join('output', 'HumanCrossModality',  f'3DdefaultParameters_B2toB5_{legacy}')
    datadir = os.path.join('..', 'data', '3D_cross_B2ToB5')

    os.makedirs(savedir, exist_ok=True)

    S, nu_S = torch.load(os.path.join(datadir, 'source_3D_B2_3regions.pt'), map_location=map_location)
    T, nu_T = torch.load(os.path.join(datadir, 'target_3D_B5_6regions.pt'), map_location=map_location)

    N = S.shape[0]
    sigmaRKHSlist,sigmaVarlist,gamma,d,labs,its,kScale,cA,cT,cS,cPi,dimEff,Csqpi,Csqlamb,eta0,lamb0,single=torch.load(os.path.join(datadir, 'parameters_3D_B2ToB5.pt'))

    
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
    print("cPi: ", cPi)
    print("dimEff: ", dimEff)
    print("Csqpi: ", Csqpi)
    print("Csqlamb: ", Csqlamb)
    print("eta0: ", eta0)
    print("lamb0: ", lamb0)
    print("single: ", single)
    print("N " + str(N))    
            
    Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist, Tlist, nu_Tlist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,gamma,d,labs,savedir,its=its,kScale=kScale,cA=cA,cT=cT,cS=cS,cPi=cPi,dimEff=d,Csqpi=Csqpi,Csqlamb=Csqlamb,eta0=eta0,lambInit=lamb0,loadPrevious=None)
    
    nu_Td = nu_Tlist[-1]
    Td = Tlist[-1]

    zeta_S = nu_S/(torch.sum(nu_S,axis=-1)[...,None])
    zeta_T = nu_T/(torch.sum(nu_T,axis=-1)[...,None])
    w_S = torch.sum(nu_S,axis=-1)[...,None]
    w_T = torch.sum(nu_T,axis=-1)[...,None]
    zeta_S[torch.squeeze(w_S == 0),...] = 0
    zeta_T[torch.squeeze(w_T == 0),...] = 0
    torch.save([S,nu_S,T,nu_T,zeta_S,zeta_T],os.path.join(savedir,'origST.pt'))
    
    gO.writeParticleVTK(S,nu_S,os.path.join(savedir, 'testOutput_S.vtk'))
    gO.writeParticleVTK(T,nu_T,os.path.join(savedir, 'testOutput_T.vtk'))
    gO.writeParticleVTK(Td,nu_Td,os.path.join(savedir, 'testOutput_Td.vtk'))

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
        gO.writeParticleVTK(D,nu_D,os.path.join(savedir, f'testOutput_D{t}.vtk'))
        gO.writeParticleVTK(D,nu_Dpi,os.path.join(savedir, f'testOutput_Dpi{t}.vtk'))
        gO.writeParticleVTK(G,nu_G,os.path.join(savedir, f'testOutput_G{t}.vtk'))
        if (t == len(Dlist) - 1):
            if not torch.is_tensor(D):
                np.savez(os.path.join(savedir, 'testOutput.npz'),D=D,nu_D=nu_D,nu_Dpi=nu_Dpi)
            else:
                torch.save([D,nu_D,nu_Dpi],os.path.join(savedir, 'testOutput.pt'))


    sigmaVar0 = sigmaVarlist[0].cpu().numpy()
    getLocalDensity(S,nu_S,sigmaVar0,os.path.join(savedir, 'density_S.vtk'),coef=2.0)
    getLocalDensity(T,nu_T,sigmaVar0,os.path.join(savedir, 'density_T.vtk'),coef=2.0)
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar0,os.path.join(savedir, 'density_D.vtk'),coef=2.0)
    
    jFile = gO.getJacobian(Dlist[-1],nu_S,nu_Dlist[-1],os.path.join(savedir, 'testOutput_D10_jacobian.vtk'))
    gO.splitZs(T,nu_T,Dlist[-1],nu_DPilist[-1],os.path.join(savedir, 'testOutput_Dpi10'),units=15,jac=jFile)
    
    S = S.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    if torch.is_tensor(Dlist[-1]):
        D = Dlist[-1].detach().cpu().numpy()
        nu_D = nu_Dlist[-1].detach().cpu().numpy()
    else:
        D = Dlist[-1]
        nu_D = nu_Dlist[-1]
        
    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))


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
    gO.writeParticleVTK(x,qw0,os.path.join(savedir, 'testOutput_D_ATau.vtk'))
    getLocalDensity(x,nu_S,sigmaVar0,os.path.join(savedir, 'density_D_ATau.vtk'),coef=2.0)
    
    sys.stdout = original
    return

if __name__ == "__main__":
    main()