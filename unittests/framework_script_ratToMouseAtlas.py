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

def saveAtlas(qx1,qw1,zeta_S,pi_ST,lamb,s,m):
    # rescale state
    D = resizeData(qx1,s,m)
    
    # compute geometrically deformed only
    nu_D = (torch.squeeze(qw1)[...,None])*zeta_S
    nu_Dpi = nu_D@pi_ST
    
    sW = dataloss.supportWeight(qx1,lamb)
    jac = getJacobian(D,nu_S,nu_D)
    
    writeParticleVTK(D, nu_D, os.path.join(savedir,"atlas_nu_D.vtk"), norm=True, condense=False, featNames=None, sW=None)
    writeParticleVTK(D, nu_Dpi, os.path.join(savedir,"atlas_nu_Dpi.vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "D": D,
        "nu_D": nu_D,
        "nu_Dpi": nu_Dpi,
        "sW": sW,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"atlas_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_D.cpu().numpy(),axis=-1),
        np.argmax(nu_D.cpu().numpy(),axis=-1)+1,
        np.sum(nu_Dpi.cpu().numpy(),axis=-1),
        np.argmax(nu_Dpi.cpu().numpy(),axis=-1)+1,
        np.squeeze(sW.cpu().numpy()),
        np.squeeze(jac.cpu().numpy()),
        np.squeeze(getEntropy(nu_Dpi.cpu().numpy())),
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "Weight_TargetFeaturs",
        "MaxVal_TargetFeatures",
        "SupportWeight_TargetSpace",
        "Jacobian",
        "Entropy"
    ]
    writeVTK(D, imageVals, imageNames, os.path.join(savedir,"atlas_deformationSummary.vtk"))
    return
                   
def saveTarget(Td, w_Td, zeta_T,s,m):
    # rescale state
    Td = resizeData(Td,s,m)
    
    # compute deformed features
    nu_Td = torch.squeeze(w_Td)[...,None]*zeta_T
    
    jac = getJacobian(Td,nu_T,nu_Td)
    
    writeParticleVTK(Td, nu_Td, os.path.join(savedir,"target_nu_Td.vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "Td": Td,
        "nu_Td": nu_Td,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"target_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_Td.cpu().numpy(),axis=-1),
        np.argmax(nu_Td.cpu().numpy(),axis=-1)+1,
        np.squeeze(jac.cpu().numpy()),
        np.squeeze(getEntropy(nu_Td.cpu().numpy())),
    ]
    imageNames = [
        "Weight_TargetFeatures",
        "MaxVal_TargetFeatures",
        "Jacobian",
        "Entropy"
    ]
    writeVTK(Td, imageVals, imageNames, os.path.join(savedir,"target_deformationSummary.vtk"))
    return


def saveOriginal(S,nu_S,T,nu_T):
    print("saving original with sizes")
    wS = torch.sum(nu_S,axis=-1)
    wT = torch.sum(nu_T,axis=-1)
    print(S.shape)
    print(nu_S.shape)
    print(T.shape)
    print(nu_T.shape)
    maxS = torch.argmax(nu_S,axis=-1)+1.0
    maxT = torch.argmax(nu_T,axis=-1)+1.0
    
    writeVTK(S,[wS.cpu().numpy(),maxS.cpu().numpy()],['Weight','Max_Feat'],os.path.join(savedir,"originalAtlas.vtk"))
    writeVTK(T,[wT.cpu().numpy(),maxT.cpu().numpy()],['Weight','Max_Feat'],os.path.join(savedir,"originalTarget.vtk"))
    return


# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
savedir = os.path.join("output", "RatToMouse", f"2DdefaultParameters","Testing")
datadir = os.path.join("..", "data", "2D_cross_ratToMouseAtlas")

os.makedirs(savedir, exist_ok=True)

S, nu_S = torch.load(os.path.join(datadir, "source_2D_ratAtlas.pt"), map_location=xmodmap.map_location)
T, nu_T = torch.load(    os.path.join(datadir, "target_2D_mouseAtlas.pt"), map_location=xmodmap.map_location)

saveOriginal(S,nu_S,T,nu_T)

d = 3
dimEff = 2
labs = 34  # in target
labS = 114  # template
sigmaRKHS = [0.2, 0.1, 0.05]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 0
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
lambInit = torch.tensor(0.4)
single = False
gamma = 0.1 #0.01 #10.0

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
dataloss = xmodmap.distance.LossVarifoldNormBoundary(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit, lambInit)
dataloss.weight = 1.

## Support Regularization
lambLoss = xmodmap.distance.SupportRestrictionReg(eta0)
lambLoss.weight = gamma

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)
piLoss.weight = gamma * 0.1 / torch.log(torch.tensor(nu_T.shape[-1]))

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=2, single=False)
hamiltonian.weight = gamma
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

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi,
    "lamb": Csqlamb
}

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
#loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(steps)

# Example of resuming == equivalent of loss.optimize(3)
'''
loss2 = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss2.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss2.optimize(2)
'''

# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

px1,pw1,qx1,qw1 = shooting(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])[-1] # get Deformed Atlas


shootingBack = xmodmap.deformation.ShootingBackwards(sigmaRKHS,Stilde,cA=cA,cS=cS,cT=cT, dimEff=dimEff, single=False)
_,_,_,_,Td,wTd = shootingBack(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target

# Test reshot atlas 
_,_,_,_,qx0,qw0 = shootingBack(px1, pw1, qx1, qw1, torch.clone(qx1.detach()), torch.clone(qw1.detach()))[-1]
qx1r = resizeData(qx0,s,m)
writeVTK(qx1r,[np.ones((qx1r.shape[0],1))],['posOfOrigAtlas'],os.path.join(savedir,'rereshot_origAtlas.vtk'))



saveAtlas(qx1.detach(),qw1.detach(),zeta_S,precondVar['pi_ST'].detach()**2,precondVar['lamb'].detach()**2,s,m)
saveTarget(Td.detach(),wTd.detach(),zeta_T,s,m)

f = loss.print_log()
f.savefig(os.path.join(savedir,"loss.png"),dpi=300)

f = loss.print_log(logScale=True)
f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)



