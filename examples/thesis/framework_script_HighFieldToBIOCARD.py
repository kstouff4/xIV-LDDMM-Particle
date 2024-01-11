'''
Author: Kaitlin Stouffer
Purpose:
    - single modality mapping (high field segmentations to low field segmentations)
    - no boundary estimation or density support weights used 
    - parameters are only rigid (A + tau + scale (isotropic)) and diffeomorphism
    
Use:
    python3 framework_script_HighFieldToBIOCARD atlas_name target_name
'''
import os
import sys
from sys import path as sys_path
sys_path.append("..")
sys_path.append("../..")
import glob

import torch
import xmodmap
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

from xmodmap.preprocess.preprocess import resizeData
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

def saveAtlas(qx1,qw1,zeta_S,s,m):
    # rescale state
    D = resizeData(qx1,s,m)
    
    # compute geometrically deformed only
    nu_D = (torch.squeeze(qw1)[...,None])*zeta_S
    jac = getJacobian(D,nu_S,nu_D)
    
    writeParticleVTK(D, nu_D, os.path.join(savedir,"atlas_nu_D.vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "D": D,
        "nu_D": nu_D,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"atlas_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_D.cpu().numpy(),axis=-1),
        np.argmax(nu_D.cpu().numpy(),axis=-1)+1,
        np.squeeze(jac.cpu().numpy()),
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "Jacobian",
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
    
    writeVTK(S,[wS.cpu().numpy(),maxS.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalAtlas.vtk"))
    writeVTK(T,[wT.cpu().numpy(),maxT.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalTarget.vtk"))
    return

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading: 8-876; 9-856; 10-836; 11-816; 12-796
letterCode = sys.argv[1] # i.e. ABEBER_150318
aFile = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/ExVivoHuman11T/HighFieldSegs/brain5_ds5.npz"

tFile = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/BIOCARD_Final/' + letterCode.split('_')[0] + '/' + letterCode + '.npz'

savedir = os.path.join("output", "ExVivoToBIOCARD", "Brain5Atlas_ShiftedAll",letterCode) # separate folder for each subject + date 

os.makedirs(savedir, exist_ok=True)
print(torch.get_default_dtype)

d = 3
dimEff = 3
sigmaRKHS = [0.1, 0.01] #[0.2,0.1,0.05] as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.35, 0.2, 0.05, 0.01] #[0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 100
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.0
cT = 1.0  # original is 0.5
cS = 10.0
single = True
gamma = 0.01

S,nu_S = getFromFile(aFile)
T,nu_T = getFromFile(tFile) # 1 = cell type (cluster); 2 = cell index; 3 = genes, 4 = regions
    
labs = nu_T.shape[-1]  # in target
labS = nu_S.shape[-1]  # template

A180 = torch.eye(3)
A180[0,0] = -1.0
A180[1,1] = -1.0
S = S@A180

# center template and target around center of mass (challenging for reshooting though)
comT = torch.mean(T,axis=0)
comT = comT
T = T - comT
comS = torch.mean(S,axis=0)
comS = comS
S = S - comS

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
) = makePQ(S, nu_S, T, nu_T)

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
    "comT": comT,
    "comS": comS
}

torch.save(sm,os.path.join(savedir,"sm.pt"))


# Model Setup

## Varifold distance without boundaries
dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, torch.eye(zeta_S.shape[-1])) # single modality with pi as identity 
dataloss.weight = 1.

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)


# Optimization

variable_init = {
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "zeta_S": zeta_S,
}

variable_to_optimize = ["px", "pw"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
}

loss = xmodmap.model.SingleModality(hamiltonian, shooting, dataloss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
#loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(steps)

# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

px1,pw1,qx1,qw1 = shooting(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])[-1] # get Deformed Atlas


shootingBack = xmodmap.deformation.ShootingBackwards(sigmaRKHS,Stilde,cA=cA,cS=cS,cT=cT, dimEff=dimEff, single=single)
_,_,_,_,Td,wTd = shootingBack(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target


saveAtlas(qx1.detach(),qw1.detach(),zeta_S,s,m)
saveTarget(Td.detach(),wTd.detach(),zeta_T,s,m)

f = loss.print_log()
f.savefig(os.path.join(savedir,"loss.png"),dpi=300)

f = loss.print_log(logScale=True)
f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)


