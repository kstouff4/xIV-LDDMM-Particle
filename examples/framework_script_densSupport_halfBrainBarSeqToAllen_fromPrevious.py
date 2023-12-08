import os
import sys
from sys import path as sys_path
sys_path.append("..")

import torch
import xmodmap
import scipy as sp
from scipy import io
import numpy as np
from matplotlib import pyplot as plt

from xmodmap.preprocess.preprocess import resizeData
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

'''
Description:
    Use to continue optimization scheme with SAME source and target but different kernels of data attachment and RKHS
    * approximates a multi-scale implementation in a sequential way 
    
    Consistencies: assumes preconditioning coefficients are the same (kScale, dimEff, w_S, Csqpi, Csqlamb)
'''

def saveAtlas(qx1,qw1,zeta_S,pi_ST,lamb,s,m):
    # rescale state
    D = resizeData(qx1,s,m)
    
    # compute geometrically deformed only
    nu_D = (torch.squeeze(qw1)[...,None])*zeta_S
    nu_Dpi = nu_D@pi_ST
    
    sW = dataloss.supportWeight(qx1,lamb)
    dW = dataloss.densityWeight(qx1)
    jac = getJacobian(D,nu_S,nu_D)
    
    writeParticleVTK(D, nu_D, os.path.join(savedir,"atlas_nu_D.vtk"), norm=True, condense=False, featNames=None, sW=None)
    writeParticleVTK(D, nu_Dpi, os.path.join(savedir,"atlas_nu_Dpi.vtk"), norm=True, condense=False, featNames=regionTypeNames, sW=None)
    summary = {
        "D": D,
        "nu_D": nu_D,
        "nu_Dpi": nu_Dpi,
        "sW": sW,
        "dW": dW,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"atlas_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_D.cpu().numpy(),axis=-1),
        np.argmax(nu_D.cpu().numpy(),axis=-1)+1,
        np.sum(nu_Dpi.cpu().numpy(),axis=-1),
        np.argmax(nu_Dpi.cpu().numpy(),axis=-1)+1,
        np.squeeze(sW.cpu().numpy()),
        np.squeeze(dW.cpu().numpy()),
        np.squeeze(jac.cpu().numpy()),
        np.squeeze(getEntropy(nu_Dpi.cpu().numpy())),
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "Weight_TargetFeaturs",
        "MaxVal_TargetFeatures",
        "SupportWeight_TargetSpace",
        "DensityWeight_TargetSpace",
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
    wS = torch.sum(nu_S,axis=-1)
    wT = torch.sum(nu_T,axis=-1)
    maxS = torch.argmax(nu_S,axis=-1)+1.0
    maxT = torch.argmax(nu_T,axis=-1)+1.0
    
    writeVTK(S,[wS.cpu().numpy(),maxS.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalAtlas.vtk"))
    writeVTK(T,[wT.cpu().numpy(),maxT.cpu().numpy()],['Weight','Max_CellType'],os.path.join(savedir,"originalTarget.vtk"))
    return

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)

dataDir = sys.argv[1] # pass in the data directory as the second argument here (as string, e.g. "D076_1L")

# Data Loading
savedir = os.path.join("output", "BarSeq", "HalfBrains","200umTo200um_Density_smallRKHSPrev", dataDir )
aFile = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/approx200um_flipZ.npz"
prevDir = os.path.join("output", "BarSeq", "HalfBrains","200umTo200um_Density", dataDir )
tFile = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeq/HalfBrains/' + dataDir + '/approx/0.2/allCombined_optimal_all.npz'

origDataFP = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/HalfBrains/' + dataDir + '/'
pref = 'filt_neurons_' + dataDir + '_CCFv2_newtypes'

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
sigmaRKHS = [0.05,0.01]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 150
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
densSigma = 0.1

saveOriginal(S,nu_S,T,nu_T)

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
    "densSigma": densSigma,
    "sigmaVar": sigmaVar,
    "gamma": gamma,
    "Csqpi": Csqpi,
    "Csqlamb": Csqlamb
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

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi,
    "lamb": Csqlamb
}

previous_variables = torch.load(os.path.join(prevDir,'precondVar.pt'))
print(previous_variables)
previous_variables_initF = lambda x: {key: x[key] / precond[key] for key in precond}
previous_variables_init = previous_variables_initF(previous_variables)
print(previous_variables)


variable_init = {
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "px": previous_variables_init["px"].clone().detach().requires_grad_(True),
    "pw": previous_variables_init["pw"].clone().detach().requires_grad_(True),
    "pi_ST": previous_variables_init["pi_ST"].clone().detach().requires_grad_(True),
    "zeta_S": zeta_S,
    "lamb": previous_variables_init["lamb"].clone().detach().requires_grad_(True)
}

variable_to_optimize = ["px", "pw", "pi_ST", "lamb"]



# check
print("qx size: ", variable_init["qx"].shape)
print("qw size: ", variable_init["qw"].shape)
print("px size: ", variable_init["px"].shape)
print("pw size: ", variable_init["pw"].shape)
print("pi_ST size: ", variable_init["pi_ST"].shape)
print("zeta_S size: ", variable_init["zeta_S"].shape)

# Example of starting from scratch

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
loss.optimize(steps)
'''
# Example of resuming == equivalent of loss.optimize(3)

loss = xmodmap.model.CrossModalityBoundary(hamiltonian, shooting, dataloss, piLoss, lambLoss)
loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(0)
'''

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


