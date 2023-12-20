import os
from sys import path as sys_path
sys_path.append("..")

import torch
import xmodmap
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

from xmodmap.preprocess.preprocess import resizeData
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
savedir = os.path.join("output", "BarSeq", "Half_Brain_D079","2D_436To26_Density_cs15_densSig0.15")
aFile = os.path.join("..","data","2DhalfBrainBarSeqToAllen") + "slice436.npz"
tFile = os.path.join("..","data","2DhalfBrainBarSeqToAllen") + "filt_neurons_D079_3L_goodsubset_CCFv2_cellSlice26.npz"

os.makedirs(savedir, exist_ok=True)

S,nu_S = getFromFile(aFile)
T,nu_T = getFromFile(tFile)

d = 2
dimEff = 2
labs = nu_T.shape[-1]  # in target
labS = nu_S.shape[-1]  # template
sigmaRKHS = [0.2, 0.1, 0.05]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 180
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.0
cT = 1.0  # original is 0.5
cS = 20.0
Csqpi = 1000.0
Csqlamb = 100.0
eta0 = torch.tensor(0.2).sqrt()
lambInit = torch.tensor(0.4)
single = False
gamma = 0.3 #0.01 #10.0
densSigma = 0.15

# make them both in z = 0 plane
if dimEff == 2:
    S[:,-1] = 0
    T[:,-1] = 0
if d == 2:
    S = S[:,0:2]
    T = T[:,0:2]


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
sm = {
    "s": s,
    "m": m,
    "cA": cA,
    "cT": cT,
    "cS": cS,
    "sigmaRKHS": sigmaRKHS,
    "d": d,
    "dimEff": dimEff,
    "single": single,
    "densSigma": densSigma
}

torch.save(sm,os.path.join(savedir,"sm.pt"))


# Model Setup

## Varifold distance with boundaries
dataloss = xmodmap.distance.LossVarifoldNormBoundary_Partial(sigmaVar, w_T, zeta_T, Ttilde, densSigma)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit, lambInit)
dataloss.weight = 1.

## Support Regularization
lambLoss = xmodmap.distance.SupportRestrictionReg(eta0)
lambLoss.weight = gamma

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)
piLoss.weight = gamma * 0.1 / torch.log(torch.tensor(nu_T.shape[-1])) #cPi

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=False)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=False)

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

variable_to_optimize = ["px", "pw", "pi_ST"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi
}

# Example of starting from scratch

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
loss.optimize(steps)

# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

