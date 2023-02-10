#from pykeops_adapted_hamiltonianLDDMM import *
from fromScratchHamiltonian import *

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
    labs = 30
    sigmaRKHS = 2.0
    sigmaVar = 2.0
    its = 10
    alpha = 3.0
    beta = 1.0
        
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: " + str(sigmaRKHS))
    print("sigmaVar: " + str(sigmaVar))
    print("its: " + str(its))
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))

    # make test dataset -- translation only 
    x = torch.arange(-50,51).type(dtype)
    X,Y,Z = torch.meshgrid(x,x,x,indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    N = S.shape[0]
    print("N " + str(N))
    nu_S = np.zeros((N,labs))
    nu_S[np.arange(N),np.random.randint(low=0,high=labs,size=N)] = 1.0 # discrete particles 
    nu_S = torch.tensor(nu_S).type(dtype)
    T = (torch.clone(S)*alpha).type(dtype)
    nu_T = (torch.clone(nu_S)*(alpha)**d).type(dtype)
    
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/output_dl_sig_its_albe_N-' + str(d) + str(labs) + '_' + str(sigmaRKHS) + str(sigmaVar) + '_' + str(its) + '_' + str(alpha) + str(beta) + '_' + str(N) + '/'
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)

    Dlist, nu_Dlist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),d,labs,savedir,its=its,beta=beta)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_S,axis=-1), np.argmax(nu_S,axis=-1)]
    imageValsT = [np.sum(nu_T,axis=-1), np.argmax(nu_T,axis=-1)]

    for i in range(labs):
        imageNames.append('feature' + str(i))
        imageValsS.append(nu_S[:,i])
        imageValsT.append(nu_T[:,i])

    vtf.writeVTK(S,imageValsS,imageNames,savedir+'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(T,imageValsT,imageNames,savedir+'testOutput_T.vtk',polyData=None)
    pointList = np.zeros((S.shape[0]*len(Dlist),d))
    polyList = np.zeros((S.shape[0]*(len(Dlist)-1),3))
    polyList[:,0] = 2
    
    for t in range(len(Dlist)):
        D = Dlist[t]
        nu_D = nu_Dlist[t]
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        for i in range(labs):
            imageValsD.append(nu_D[:,i])
        vtf.writeVTK(D,imageValsD,imageNames,savedir+'testOutput_D' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez(savedir+'testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)
        else:
            pointList[int(t*len(D)):int((t+1)*len(D))] = D
            polyList[int(t*len(D)):int((t+1)*len(D)),1] = np.arange(t*len(D),(t+1)*len(D))
            polyList[int(t*len(D)):int((t+1)*len(D)),2] = np.arange((t+1)*len(D),(t+2)*len(D))
    #polyList[:,1] = polyList[:,1]+1
    #polyList[:,2] = polyList[:,2]+1
    vtf.writeVTK(pointList,[],[],savedir+'testOutput_curves.vtk',polyData=polyList)
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
    return

if __name__ == "__main__":
    main()