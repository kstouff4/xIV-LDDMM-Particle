from fromScratchHamiltonianAT import *
from analyzeOutput import *

import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # create test dataset
    d = 3
    labs = 2
    sigmaRKHS = 8.0
    sigmaVar = 2.0
    its = 5
    alpha = 1.0
    beta = 1.0
    alphaA = 0.5
    gamma = 0.5
    
    original = sys.stdout

    # make test dataset -- translation only 
    x = torch.arange(-10,11).type(dtype)
    X,Y,Z = torch.meshgrid(x,x,x,indexing='ij')
    nu_S = torch.zeros((X.shape[0],X.shape[1],X.shape[2],labs)).type(dtype)
    nu_S[:,:,0:10,0] = 0.8
    nu_S[:,:,0:10,1] = 0.2
    nu_S[:,:,10:,0] = 0.2
    nu_S[:,:,10:,1] = 0.8
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    N = S.shape[0]
    T = (torch.clone(S)*alpha).type(dtype)
    nu_T = (torch.clone(nu_S)*(alpha)**d).type(dtype)
    nu_T[:,:,0:5,0] = 0.8
    nu_T[:,:,0:5,1] = 0.2
    nu_T[:,:,5:,0] = 0.2
    nu_T[:,:,5:,1] = 0.8
    
    nu_S = torch.stack((nu_S[...,0].flatten(),nu_S[...,1].flatten()),axis=-1).type(dtype)
    nu_T = torch.stack((nu_T[...,0].flatten(),nu_T[...,1].flatten()),axis=-1).type(dtype)
    
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/output_dl_sig_its_albe_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(alpha) + str(beta) + '_' + str(N) + str(alphaA) + str(gamma) + '/'
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
    print("alphaA: ", alphaA)
    print("gamma: ", gamma)
    
    print("N " + str(N))
    
    Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),torch.tensor(alphaA).type(dtype),torch.tensor(gamma).type(dtype),d,labs,savedir,its=its,beta=beta)
    
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