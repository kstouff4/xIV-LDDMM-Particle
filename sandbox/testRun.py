from pykeops_adapted_hamiltonianLDDMM import *

import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

def main():
    # create test dataset
    N = 10
    d = 3
    labs = 4
    sigma = 0.5

    # make test dataset 
    S = torch.rand(N,d) # coordinates 
    nu_S = np.zeros((N,labs))
    nu_S[np.arange(N),np.random.randint(low=0,high=labs,size=N)] = 1.0 # discrete particles 
    nu_S = torch.tensor(nu_S)
    T = torch.rand(N,d)
    nu_T = np.zeros((N,labs))
    nu_T[np.arange(N),np.random.randint(low=0,high=labs,size=N)] = 1.0 
    nu_T = torch.tensor(nu_T)
    
    D, nu_D = callOptimize(S,nu_S,T,nu_T,sigma,d,labs)
    
    S=S.detach().cpu().numpy()
    T=T.detach().cpu().numpy()
    nu_S = w_S.detach().cpu().numpy()*zeta_S.detach().cpu().numpy()
    nu_T = w_T.detach().cpu().numpy()*zeta_T.detach().cpu().numpy()

    np.savez('testOutput.npz',S=S, nu_S=nu_S,T=T,nu_T=nu_T,D=D,nu_D=nu_D)

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [w_S, np.argmax(nu_S,axis=-1)]
    imageValsT = [w_T, np.argmax(nu_T,axis=-1)]
    imageValsD = [w_D, np.argmax(nu_D,axis=-1)]

    for i in range(labs):
        imageNames.append('feature' + str(i))
        imageValsS.append(nu_S[:,i])
        imageValsT.append(nu_T[:,i])
        imageValsD.append(nu_D[:,i])

    vtf.writeVTK(S,imageValsS,imageNames,'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(T,imageValsT,imageNames,'testOutput_T.vtk',polyData=None)
    vtf.writeVTK(D,imageValsD,imageNames,'testOutput_D.vtk',polyData=None)
    
    return

if __name__ == "__main__":
    main()