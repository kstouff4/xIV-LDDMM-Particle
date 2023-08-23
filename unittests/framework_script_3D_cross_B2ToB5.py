import os
from sys import path as sys_path
sys_path.append("..")

import torch
import xmodmap

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
savedir = os.path.join("output", "HumanCrossModality", f"3DdefaultParameters_B2toB5")
datadir = os.path.join("..", "data", "3D_cross_B2ToB5")

os.makedirs(savedir, exist_ok=True)

S, nu_S = torch.load(os.path.join(datadir, "source_3D_B2_3regions.pt"), map_location=xmodmap.map_location)
T, nu_T = torch.load(    os.path.join(datadir, "target_3D_B5_6regions.pt"), map_location=xmodmap.map_location)

d = 3
dimEff = 3
single = True
labs = 2  # in target
labS = 28  # template
sigmaRKHS = [0.2, 0.1, 0.05]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
its = 100
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.0
cT = 1.0  # original is 0.5
cS = 10.0
Csqpi = 100.0
Csqlamb = 100.0
eta0 = torch.sqrt(torch.tensor(0.2))
lamb0 = torch.tensor(0.4)
lambInit = -1.

gamma = 1.0  # 0.01 #10.0


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
           Csqpi=Csqpi,
           lambInit=lambInit,
           Csqlamb=Csqlamb)

# Model Setup

## Varifold distance with boundaries
dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
dataloss.weight = 1.

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)
piLoss.weight = gamma * 0.1 / torch.log(torch.tensor(nu_T.shape[-1]))

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)


# Optimization

variable_init = {
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "pi_ST": ((1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach()).requires_grad_(True),
    "zeta_S": zeta_S,
}

variable_to_optimize = ["px", "pw", "pi_ST"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi,
}

loss = xmodmap.model.CrossModality(hamiltonian, shooting, dataloss, piLoss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
#loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(11)