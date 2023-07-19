from sys import path as sys_path
sys_path.append('..')

import numpy as np
import torch


from sandbox.crossModalityHamiltonianATSCalibrated_Boundary import *
from sandbox.analyzeOutput import *

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float32"  # "float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor  # DoubleTensor
    map_location = lambda storage, loc: storage.cuda(0)
else:
    dtype = torch.FloatTensor
    map_location = torch.device('cpu')

torch.set_default_tensor_type(dtype)


def main():
    savedir = ''
    data_dir = os.path.join('..', 'data', '2D_cross_ratToMouseAtlas')

    S, nu_S = torch.load(os.path.join(data_dir, 'source_2D_ratAtlas.pt'), map_location=map_location)
    T, nu_T = torch.load(os.path.join(data_dir, 'target_2D_mouseAtlas.pt'), map_location=map_location)
    sigmaRKHSlist, sigmaVarlist, gamma, d, labs, its, kScale, cA, cT, cS, cPi, dimEff, Csqpi, Csqlamb, eta0, lamb0, single = torch.load(os.path.join(data_dir, 'parameters_2D_ratToMouseAtlas.pt'), map_location=map_location)

    Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist, Tlist, nu_Tlist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,torch.tensor(gamma).type(dtype),d,labs,savedir,its=its,kScale=torch.tensor(kScale).type(dtype),cA=torch.tensor(cA).type(dtype),cT=torch.tensor(cT).type(dtype),cS=cS,cPi=cPi,dimEff=dimEff,Csqpi=Csqpi,Csqlamb=Csqlamb,eta0=eta0,lambInit=lamb0,loadPrevious=None,single=single)


    return Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist, Tlist, nu_Tlist


if __name__ == "__main__":
    main()