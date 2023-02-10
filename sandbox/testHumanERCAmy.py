import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode/')
import estimateSubsample as ess

from fromScratchHamiltonian import *
from analyzeOutput import *

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():

    d = 3
    labs = 3
    sigmaRKHS = 8.0
    sigmaVar = 2.0
    its = 10
    alpha = 1.0
    beta = 1.0
    
    original = sys.stdout

    outpath='/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Human/'
    imgPref='/cis/home/kstouff4/Documents/datasets/BIOCARD/SubsetFall2022/Segmentations/ABEBER/'

    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 

    S,nu_XS = ess.makeAllXandZ(imgPref+'150318/AMYGDALA+ERC+TEC.img', outpath+'ABEBER_150318_', thickness=1000, res=1.0,sig=0.1,C=-1,flip=False)
    T,nu_XT = ess.makeAllXandZ(imgPref+'170831/AMYGDALA+ERC+TEC.img', outpath+'ABEBER_170831_', thickness=1000, res=1.0,sig=0.1,C=-1,flip=False)
    
    print("min and max")
    print(np.min(nu_XS))
    print(np.max(nu_XS))
    
    nu_S = torch.zeros((S.shape[0],3)).type(dtype)
    nu_S[:,nu_XS] = 1.0
    nu_T = torch.zeros((T.shape[0],3)).type(dtype)
    nu_T[:,nu_XT] = 1.0
    
    S = torch.tensor(S).type(dtype)
    T = torch.tensor(T).type(dtype)
    
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/output_dl_sig_its_albe_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(alpha) + str(beta) + '_' + str(N) + '/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    
    sys.stdout = open(savedir+'test.txt','w')
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: " + str(sigmaRKHS))
    print("sigmaVar: " + str(sigmaVar))
    print("its: " + str(its))
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))
    
    print("N " + str(N))
    
    Dlist, nu_Dlist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),d,labs,savedir,its=its,beta=beta)
    
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
    pointList = np.zeros((S.shape[0]*len(Dlist),d))
    polyList = np.zeros((S.shape[0]*(len(Dlist)-1),3))
    polyList[:,0] = 2
    
    for t in range(len(Dlist)):
        D = Dlist[t]
        nu_D = nu_Dlist[t]
        zeta_D = nu_D/(np.sum(nu_D,axis=-1)[...,None])
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        for i in range(labs):
            imageValsD.append(zeta_D[:,i])
        vtf.writeVTK(D,imageValsD,imageNames,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez(savedir+'testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)
        else:
            pointList[int(t*len(D)):int((t+1)*len(D))] = D
            polyList[int(t*len(D)):int((t+1)*len(D)),1] = np.arange(t*len(D),(t+1)*len(D))
            polyList[int(t*len(D)):int((t+1)*len(D)),2] = np.arange((t+1)*len(D),(t+2)*len(D))

    vtf.writeVTK(pointList,[],[],savedir+'testOutput_curves.vtk',polyData=polyList)
    volS = np.prod(np.max(S,axis=(0,1)) - np.min(S,axis=(0,1)))
    volT = np.prod(np.max(T,axis=(0,1)) - np.min(T,axis=(0,1)))
    volD = np.prod(np.max(Dlist[-1],axis=(0,1)) - np.min(Dlist[-1],axis=(0,1)))
    getLocalDensity(S,nu_S,sigmaVar,savedir+'density_S.vtk')
    getLocalDensity(T,nu_T,sigmaVar,savedir+'density_T.vtk')
    getLocalDensity(Dlist[-1],nu_Dlist[-1],sigmaVar,savedir+'density_D.vtk')
    
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
    
    sys.stdout = original
    return

if __name__ == "__main__":
    main()