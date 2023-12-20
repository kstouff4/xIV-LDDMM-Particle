import os
import sys
from sys import path as sys_path
sys_path.append("..")

import torch
import xmodmap

from xmodmap.preprocess.preprocess import resizeData
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)

def saveAtlas(qx1,qw1,zeta_S,pi_ST,s,m):
    # rescale state
    D = resizeData(qx1,s,m)
    
    # compute geometrically deformed only
    nu_D = (torch.squeeze(qw1)[...,None])*zeta_S
    nu_Dpi = nu_D@pi_ST
    
    jac = getJacobian(D,nu_S,nu_D)
    
    writeParticleVTK(D, nu_D, os.path.join(savedir,"atlas_nu_D.vtk"), norm=True, condense=False, featNames=None, sW=None)
    writeParticleVTK(D, nu_Dpi, os.path.join(savedir,"atlas_nu_Dpi.vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "D": D,
        "nu_D": nu_D,
        "nu_Dpi": nu_Dpi,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"atlas_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_D.cpu().numpy(),axis=-1),
        np.argmax(nu_D.cpu().numpy(),axis=-1)+1,
        np.sum(nu_Dpi.cpu().numpy(),axis=-1),
        np.argmax(nu_Dpi.cpu().numpy(),axis=-1)+1,
        np.squeeze(jac.cpu().numpy()),
        np.squeeze(getEntropy(nu_Dpi.cpu().numpy())),
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "Weight_TargetFeaturs",
        "MaxVal_TargetFeatures",
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
    
    writeVTK(S,[wS.cpu().numpy(),maxS.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalAtlas.vtk"))
    writeVTK(T,[wT.cpu().numpy(),maxT.cpu().numpy()],['Weight','Max_Bin'],os.path.join(savedir,"originalTarget.vtk"))
    return



# Data Loading
savedir = os.path.join("output", "HumanCrossModality", f"3DdefaultParameters_B2toB5_multiScale")
datadir = os.path.join("..", "data", "3D_cross_B2ToB5")

os.makedirs(savedir, exist_ok=True)

S, nu_S = torch.load(os.path.join(datadir, "source_3D_B2_3regions.pt"), map_location=xmodmap.map_location)
T, nu_T = torch.load(os.path.join(datadir, "target_3D_B5_6regions.pt"), map_location=xmodmap.map_location)


d = 3
dimEff = 3
single = True
labs = 2  # in target
labS = 28  # template
sigmaRKHS = [0.5, 0.2, 0.1, 0.05]  # [0.2,0.1,0.05] # as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
its = 100
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.
cT = 1.  # original is 0.5
cS = 10.0
Csqpi = 100.0
Csqlamb = 100.0
eta0 = torch.sqrt(torch.tensor(0.2))
lamb0 = torch.tensor(0.4)
lambInit = -1.

gamma = 1.0  # 0.01 #10.0

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

torch.save(sm,os.path.join(savedir,"sm.pt"))

# Model Setup

## Varifold distance with boundaries
dataloss = xmodmap.distance.LossVarifoldNormWithRigidCrossMod(sigmaVar, w_T,  zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
dataloss.weight = 1.

## Varifold distance with boundaries
datalossR = xmodmap.distance.LossVarifoldNormWithRigid([0.02], w_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
datalossR.normalize_across_scale(Stilde, w_S)
datalossR.weight = 1.

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)
piLoss.weight = gamma * 0.1 / torch.log(torch.tensor(nu_T.shape[-1]))


hamiltonian = xmodmap.deformation.HamiltonianGS(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.ShootingWithRigid(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)


# Optimization

variable_init = {
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "pi_ST": ((1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach()).requires_grad_(True),
    "zeta_S": zeta_S,
    "pR": torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]).requires_grad_(True),
    "R": torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]).requires_grad_(True),
    "ptau": torch.tensor([0.,0.,0.]).requires_grad_(True),
    "tau": torch.tensor([0.,0.,0.]).requires_grad_(True),
    "palpha" : torch.tensor(0.).requires_grad_(True),
    "alpha" : torch.tensor(1.).requires_grad_(True)
}

variable_to_optimize = ["px", "pw", "pi_ST", "pR", "ptau", "palpha"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
    "pi_ST": Csqpi,
    "pR": 20.*torch.rsqrt(kScale),
    "ptau": 20.*torch.rsqrt(kScale),
    "palpha" : 20.*torch.rsqrt(kScale)
}


loss = xmodmap.model.CrossModalityGS(hamiltonian, shooting, dataloss, datalossR, piLoss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
#loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(200)

precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

pxo = loss.precond(loss.variables)["px"]
pwo = loss.precond(loss.variables)["pw"]
pRo = loss.precond(loss.variables)["pR"]
ptauo = loss.precond(loss.variables)["ptau"]
palphao = loss.precond(loss.variables)["palpha"]

(_, _, _, qx1, qw1, tau1), _, R1, _, alpha1 = shooting(pxo, pwo, pRo, ptauo, palphao, variable_init["qx"], variable_init["qw"], variable_init["R"], variable_init["tau"], variable_init["alpha"])[-1]
xc = (qw1 * qx1).sum(dim=0) / (qw1.sum(dim=0))

nqx1 = qx1 - xc
nqx1 =  nqx1.reshape((*nqx1.shape,1))
nqx1 = 1/alpha1 * R1.T @ nqx1 
nqx1 = nqx1.squeeze(-1) - tau1 + xc

saveAtlas(nqx1.detach(),((1.0/alpha1) * qw1).detach(),zeta_S,precondVar['pi_ST'].detach()**2,s,m)
'''
ending = xmodmap.preprocess.preprocess.resizeData(nqx1, s, m)
nu_ending = 1/alpha1*(qw1 * zeta_S) @ loss.precond(loss.variables)["pi_ST"]**2
xmodmap.io.getOutput.writeParticleVTK(ending.detach(), nu_ending.detach(), savedir+"atlasDeformed.vtk")
'''