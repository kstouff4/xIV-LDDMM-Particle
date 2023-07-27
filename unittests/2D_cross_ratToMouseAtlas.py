import sys
from sys import path as sys_path

sys_path.append("..")

import torch

legacy = False

if legacy:
    from sandbox.crossModalityHamiltonianATSCalibrated_Boundary_legacy import *
else:
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
    map_location = torch.device("cpu")

torch.set_default_tensor_type(dtype)

# set random seed
torch.manual_seed(0)
np.random.seed(0)


def main():
    savedir = os.path.join("output", "RatToMouse", f"2DdefaultParameters_{legacy}")
    datadir = os.path.join("..", "data", "2D_cross_ratToMouseAtlas")

    os.makedirs(savedir, exist_ok=True)

    S, nu_S = torch.load(
        os.path.join(datadir, "source_2D_ratAtlas.pt"), map_location=map_location
    )
    T, nu_T = torch.load(
        os.path.join(datadir, "target_2D_mouseAtlas.pt"), map_location=map_location
    )
    (
        sigmaRKHSlist,
        sigmaVarlist,
        gamma,
        d,
        labs,
        its,
        kScale,
        cA,
        cT,
        cS,
        cPi,
        dimEff,
        Csqpi,
        Csqlamb,
        eta0,
        lamb0,
        single,
    ) = torch.load(
        os.path.join(datadir, "parameters_2D_ratToMouseAtlas.pt"),
        map_location=map_location,
    )

    N = S.shape[0]

    Dlist, nu_Dlist, nu_DPilist, Glist, nu_Glist, Tlist, nu_Tlist = callOptimize(
        S,
        nu_S,
        T,
        nu_T,
        sigmaRKHSlist,
        sigmaVarlist,
        gamma,
        d,
        labs,
        savedir,
        its=11,
        kScale=kScale,
        cA=cA,
        cT=cT,
        cS=cS,
        cPi=cPi,
        dimEff=dimEff,
        Csqpi=Csqpi,
        Csqlamb=Csqlamb,
        eta0=eta0,
        lambInit=lamb0,
        loadPrevious=None,
        single=single,
    )

    print("Dlist", Dlist[-1])
    print("nu_DPilist", nu_DPilist[-1])

    return


if __name__ == "__main__":
    main()
