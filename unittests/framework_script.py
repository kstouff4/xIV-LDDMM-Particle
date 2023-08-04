import os
from sys import path as sys_path
sys_path.append("..")

import torch
import xmodmap

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
savedir = os.path.join("output", "RatToMouse", f"2DdefaultParameters")
datadir = os.path.join("..", "data", "2D_cross_ratToMouseAtlas")

os.makedirs(savedir, exist_ok=True)

S, nu_S = torch.load(os.path.join(datadir, "source_2D_ratAtlas.pt"), map_location=xmodmap.map_location)
T, nu_T = torch.load(    os.path.join(datadir, "target_2D_mouseAtlas.pt"), map_location=xmodmap.map_location)

from xmodmap.preprocess.makePQ_legacy import makePQ
(
    w_S,
    w_T,
    zeta_S,
    zeta_T,
    q0,
    p0,
    numS,
    Stilde,
    Ttilde,
    s,
    m,
    pi_STinit,
    lamb0,
) = makePQ(S, nu_S, T, nu_T,
           Csqpi=10000.0,
           lambInit=torch.tensor(0.4),
           Csqlamb= 100.)

d = 3
dimEff = 2
labs = 34  # in target
labS = 114  # template
sigmaRKHS = [0.2, 0.1, 0.05]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
its = 81
alphaSt = 'sl536'
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.0
cT = 1.0  # original is 0.5
cS = 10.0
Csqpi = 10000.0
Csqlamb = 100.0
eta0 = torch.tensor(0.2).sqrt()
lamb0 = torch.tensor(0.4)
single = False


# Model Setup

## Varifold distance with boundaries
dataloss = xmodmap.distance.LossVarifoldNormBoundary(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit, lamb0)
dataloss.weight = 10.

## Support Regularization
lambLoss = xmodmap.distance.SupportRestrictionReg()
lambLoss.weight = 1.

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)


## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=2, single=False)
hamiltonian.weight = 1.
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=2, single=False)

# Optimization

variable_init = {
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "pi_ST": ((1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach()).requires_grad_(True),
    "zeta_S": zeta_S,
    "lamb": (lamb0.sqrt().clone().detach() / Csqlamb).requires_grad_(True)
}

variable_to_optimize = ["px", "pw", "pi_ST", "lamb"]

#TODO: check if this is correct
pwTilde = torch.rsqrt(kScale) / dimEff / w_S
pxTilde = torch.rsqrt(kScale)

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss, variable_init, variable_to_optimize, pwTilde=pwTilde, pxTilde=pxTilde, Csqpi=Csqpi)
loss.optimize()

# Plotting



# Saving
