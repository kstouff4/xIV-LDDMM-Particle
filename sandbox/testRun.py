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
    N = 10
    d = 3
    labs = 4
    sigmaRKHS = 2.0
    sigmaVar = 2.0
    its = 20

    # make test dataset -- translation only 
    x = torch.arange(-2,3).type(dtype)
    X,Y,Z = torch.meshgrid(x,x,x,indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1).type(dtype)
    N = S.shape[0]
    print("number of particles is " + str(N))
    nu_S = np.zeros((N,labs))
    nu_S[np.arange(N),np.random.randint(low=0,high=labs,size=N)] = 1.0 # discrete particles 
    nu_S = torch.tensor(nu_S).type(dtype)
    T = (torch.clone(S)*1.2).type(dtype)
    nu_T = (torch.clone(nu_S)*(1.2)**d).type(dtype)
    #T = T[:-1,:] # remove one particle so is M x N
    #T[0,-1] += 0.3 # translate first particle in z
    #T[-1,-1] -= 0.2 # translate last particle in z 
    #nu_T = torch.clone(nu_S) # same exact features 
    #nu_T = np.zeros((N,labs))
    #nu_T[np.arange(N),np.random.randint(low=0,high=labs,size=N)] = 1.0 
    #nu_T = torch.tensor(nu_T)
    
    Dlist, nu_Dlist = callOptimize(S,nu_S,T,nu_T,torch.tensor(sigmaRKHS).type(dtype),torch.tensor(sigmaVar).type(dtype),d,labs,its=its)
    
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

    vtf.writeVTK(S,imageValsS,imageNames,'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(T,imageValsT,imageNames,'testOutput_T.vtk',polyData=None)
    pointList = np.zeros((S.shape[0]*len(Dlist),d))
    polyList = np.zeros((S.shape[0]*(len(Dlist)-1),3))
    polyList[:,0] = 2
    
    for t in range(len(Dlist)):
        D = Dlist[t]
        nu_D = nu_Dlist[t]
        imageValsD = [np.sum(nu_D,axis=-1), np.argmax(nu_D,axis=-1)]
        for i in range(labs):
            imageValsD.append(nu_D[:,i])
        vtf.writeVTK(D,imageValsD,imageNames,'testOutput_D' + str(t) + '.vtk',polyData=None)
        if (t == len(Dlist) - 1):
            np.savez('testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)
        else:
            pointList[int(t*len(D)):int((t+1)*len(D))] = D
            polyList[int(t*len(D)):int((t+1)*len(D)),1] = np.arange(t*len(D),(t+1)*len(D))
            polyList[int(t*len(D)):int((t+1)*len(D)),2] = np.arange((t+1)*len(D),(t+2)*len(D))
    vtf.writeVTK(pointList,[],[],'testOutput_curves.vtk',polyData=polyList)
    return

if __name__ == "__main__":
    main()