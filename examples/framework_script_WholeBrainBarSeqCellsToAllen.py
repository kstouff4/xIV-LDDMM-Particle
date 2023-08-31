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
    writeParticleVTK(D, nu_Dpi, os.path.join(savedir,"atlas_nu_Dpi.vtk"), norm=True, condense=False, featNames=regionTypeNames, sW=None)
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


# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
savedirOld = os.path.join("output", "BarSeq", "Whole_Brain_2023","200umTo200um_smallSigma")
savedir = os.path.join("output", "BarSeq", "Whole_Brain_2023","200umTo200um_smallSigmaRedo")
aFile = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/approx200um_flipZ.npz"
tFile = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeqAligned/Whole_Brain_2023/sig0.25Align_200um/Cells/all_optimal_all.npz'

origDataFP = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Whole_Brain_2023/'
pref = 'filt_neurons-clust3'

rtList = []
rt = sp.io.loadmat(origDataFP + pref + '_regionTypes.mat',appendmat=False)
regionTypeNames = rt['regionTypes']
for i in range(len(regionTypeNames)):
    rtList.append(regionTypeNames[i][0][0])
regionTypeNames = rtList

os.makedirs(savedir, exist_ok=True)
print(torch.get_default_dtype)

S,nu_S = getFromFile(aFile)
T,nu_T = getFromFile(tFile)

print("S type: ", S.dtype)
print("nu_S type: ", nu_S.dtype)
print("T type: ", T.dtype)
print("nu_T type: ", nu_T.dtype)

d = 3
dimEff = 3
labs = nu_T.shape[-1]  # in target
labS = nu_S.shape[-1]  # template
sigmaRKHS = [0.1, 0.05, 0.01] #[0.2,0.1,0.05] as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.2, 0.05, 0.02, 0.01] #[0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 200
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

sm = {
    "s": s,
    "m": m,
    "cA": cA,
    "cT": cT,
    "cS": cS,
    "sigmaRKHS": sigmaRKHS,
    "d": d,
    "dimEff": dimEff,
    "single": single
}

torch.save(sm,os.path.join(savedir,"sm.pt"))


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

variable_to_optimize = ["px", "pw", "pi_ST", "lamb"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi,
    "lamb": Csqlamb
}
'''
loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
loss.optimize(steps)
'''
# Example of resuming == equivalent of loss.optimize(3)

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss.resume(variable_init, os.path.join(savedirOld, 'checkpoint.pt'))
loss.optimize(steps)


# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

px1,pw1,qx1,qw1 = shooting(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])[-1] # get Deformed Atlas


shootingBack = xmodmap.deformation.ShootingBackwards(sigmaRKHS,Stilde,cA=cA,cS=cS,cT=cT, dimEff=dimEff, single=False)
_,_,_,_,Td,wTd = shootingBack(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target


saveAtlas(qx1.detach(),qw1.detach(),zeta_S,precondVar['pi_ST'].detach()**2,precondVar['lamb'].detach()**2,s,m)
saveTarget(Td.detach(),wTd.detach(),zeta_T,s,m)

f = loss.print_log()
f.savefig(os.path.join(savedir,"loss.png"),dpi=300)

f = loss.print_log(logScale=True)
f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)


