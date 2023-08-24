import sys
from sys import path as sys_path

sys_path.append("..")

import torch

legacy = True

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
torch.set_printoptions(precision=6)


def main():
    original = sys.stdout

    savedir = os.path.join(
        "output", "HumanCrossModality", f"3DdefaultParameters_B2toB5_{legacy}"
    )
    datadir = os.path.join("..", "data", "3D_cross_B2ToB5")

    os.makedirs(savedir, exist_ok=True)

    S, nu_S = torch.load(
        os.path.join(datadir, "source_3D_B2_3regions.pt"), map_location=map_location
    )
    T, nu_T = torch.load(
        os.path.join(datadir, "target_3D_B5_6regions.pt"), map_location=map_location
    )

    N = S.shape[0]
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
    ) = torch.load(os.path.join(datadir, "parameters_3D_B2ToB5.pt"))

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
        its=110,
        kScale=kScale,
        cA=cA,
        cT=cT,
        cS=cS,
        cPi=cPi,
        dimEff=d,
        Csqpi=Csqpi,
        Csqlamb=Csqlamb,
        eta0=eta0,
        lambInit=lamb0,
        loadPrevious=None,
        single=single
    )

    print("Dlist", Dlist[-1])
    print("nu_DPilist", nu_DPilist[-1])
    print("Glist", Glist[-1])
    print("nu_Glist", nu_Glist[-1])
    print("Tlist", Tlist[-1])
    print("nu_Tlist", nu_Tlist[-1])

    return


if __name__ == "__main__":
    main()
