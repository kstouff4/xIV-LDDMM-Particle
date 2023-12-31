import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path

sys_path.append("..")
sys_path.append("../xmodmap")
sys_path.append("../xmodmap/io")

import getInput as gi
import initialize as init

sys_path.append("/cis/home/kstouff4/Documents/SurfaceTools/")
import vtkFunctions as vtf

import torch

from fromScratchHamiltonianATCalibrated import *
from analyzeOutput import *

np_dtype = "float32"
dtype = torch.cuda.FloatTensor

import nibabel as nib

#############################################################################


def main():
    d = 3
    labs = 12
    sigmaRKHS = [0.1, 0.2]
    sigmaVar = [0.1, 0.2, 0.5]
    its = 50
    alphaSt = "B2toB5"
    beta = None
    res = 1.0
    gamma = 0.01
    kScale = 1
    extra = "newHs100Float32c20"
    cS = 20.0
    cA = 1.0
    cT = 1.0

    original = sys.stdout

    outpath = (
        "/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Human/"
    )

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    if not os.path.exists(outpath + str(alphaSt)):
        os.mkdir(outpath + str(alphaSt))

    b2 = "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain2/3DSegmentations/allMerge_smoothe.nii.gz"
    b5 = "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain5/3DSegmentations/Brain5_allMerge_smoothe.nii.gz"

    # amygdala, CA1, CA2, CA3, DG-granular, DG-hilus, DG-molecular, ERC, ERC ext, Presubiculum, Parasubiculum, Subiculum
    S, nu_S = gi.makeFromSingleChannelImage(
        b2, 0.125 * 8, bg=0, ordering=[3, 4, 5, 6, 9, 10, 11, 8, 7, 12, 13, 14], ds=8
    )
    T, nu_T = gi.makeFromSingleChannelImage(
        b5, 0.125 * 8, bg=0, ordering=[15, 2, 3, 4, 7, 9, 10, 6, 14, 11, 12, 13], ds=8
    )

    N = S.shape[0]

    savedir = (
        outpath
        + str(alphaSt)
        + "/output_dl_sig_its_be_N-"
        + str(d)
        + str(labs)
        + "_"
        + str(sigmaRKHS)
        + str(sigmaVar)
        + "_"
        + str(its)
        + "_"
        + str(beta)
        + "_"
        + str(N)
        + str(gamma)
        + extra
        + "/"
    )
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    sys.stdout = open(savedir + "test.txt", "w")
    print("Parameters")
    print("d: " + str(d))
    print("labs: " + str(labs))
    print("sigmaRKHS: " + str(sigmaRKHS))
    print("sigmaVar: " + str(sigmaVar))
    print("its: " + str(its))
    print("beta: " + str(beta))
    print("N " + str(N))

    """
    Sp=S.clone().detach().cpu().numpy()
    Tp=T.clone().detach().cpu().numpy()
    nu_Sp = nu_S.clone().detach().cpu().numpy()
    nu_Tp = nu_T.clone().detach().cpu().numpy()

    imageNames = ['weights', 'maxImageVal']
    imageValsS = [np.sum(nu_Sp,axis=-1), np.argmax(nu_Sp,axis=-1)]
    imageValsT = [np.sum(nu_Tp,axis=-1), np.argmax(nu_Tp,axis=-1)]

    zeta_Sp = nu_Sp/(np.sum(nu_Sp,axis=-1)[...,None])
    zeta_Tp = nu_Tp/(np.sum(nu_Tp,axis=-1)[...,None])
    for i in range(labs):
        imageNames.append('zeta_' + str(i))
        imageValsS.append(zeta_Sp[:,i])
        imageValsT.append(zeta_Tp[:,i])

    vtf.writeVTK(Sp,imageValsS,imageNames,savedir+'testOutput_S.vtk',polyData=None)
    vtf.writeVTK(Tp,imageValsT,imageNames,savedir+'testOutput_T.vtk',polyData=None)
    """
    Arot = torch.zeros((3, 3)).type(dtype)
    Arot[0, 0] = 1.0
    Arot[1, 1] = -1.0
    Arot[2, 2] = 1.0
    T, nu_T = init.applyAffine(T, nu_T, Arot, torch.zeros((1, d)).type(dtype))

    sigmaRKHSlist = []
    sigmaVarlist = []
    for sigg in sigmaRKHS:
        sigmaRKHSlist.append(torch.tensor(sigg).type(dtype))
    for sigg in sigmaVar:
        sigmaVarlist.append(torch.tensor(sigg).type(dtype))

    # Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(S,nu_S,T,nu_T,sigmaRKHSlist,sigmaVarlist,torch.tensor(gamma).type(dtype),d,labs,savedir,its=its,kScale=torch.tensor(kScale).type(dtype))

    Dlist, nu_Dlist, Glist, nu_Glist = callOptimize(
        S,
        nu_S,
        T,
        nu_T,
        sigmaRKHSlist,
        sigmaVarlist,
        torch.tensor(gamma).type(dtype),
        d,
        labs,
        savedir,
        its=its,
        kScale=torch.tensor(kScale).type(dtype),
        cA=torch.tensor(cA).type(dtype),
        cT=torch.tensor(cT).type(dtype),
        cS=cS,
    )

    S = S.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    imageNames = ["weights", "maxImageVal"]
    imageValsS = [np.sum(nu_S, axis=-1), np.argmax(nu_S, axis=-1)]
    imageValsT = [np.sum(nu_T, axis=-1), np.argmax(nu_T, axis=-1)]

    zeta_S = nu_S / (np.sum(nu_S, axis=-1)[..., None])
    zeta_T = nu_T / (np.sum(nu_T, axis=-1)[..., None])
    for i in range(labs):
        imageNames.append("zeta_" + str(i))
        imageValsS.append(zeta_S[:, i])
        imageValsT.append(zeta_T[:, i])

    vtf.writeVTK(S, imageValsS, imageNames, savedir + "testOutput_S.vtk", polyData=None)
    vtf.writeVTK(T, imageValsT, imageNames, savedir + "testOutput_T.vtk", polyData=None)

    pointList = np.zeros((S.shape[0] * len(Dlist), d))
    polyList = np.zeros((S.shape[0] * (len(Dlist) - 1), 3))
    polyList[:, 0] = 2

    pointListG = np.zeros((Glist[0].shape[0] * len(Glist), d))
    polyListG = np.zeros((Glist[0].shape[0] * (len(Glist) - 1), 3))
    polyListG[:, 0] = 2
    featList = np.zeros((S.shape[0] * len(Dlist), 1))
    featListG = np.zeros((Glist[0].shape[0] * len(Glist), 1))

    for t in range(len(Dlist)):
        D = Dlist[t]
        G = Glist[t]
        nu_D = nu_Dlist[t]
        nu_G = nu_Glist[t]
        zeta_D = nu_D / (np.sum(nu_D, axis=-1)[..., None])
        imageValsD = [np.sum(nu_D, axis=-1), np.argmax(nu_D, axis=-1)]
        for i in range(labs):
            imageValsD.append(zeta_D[:, i])
        vtf.writeVTK(
            D,
            imageValsD,
            imageNames,
            savedir + "testOutput_D" + str(t) + ".vtk",
            polyData=None,
        )
        vtf.writeVTK(
            G,
            [nu_G],
            ["Weights"],
            savedir + "testOutput_G" + str(t) + ".vtk",
            polyData=None,
        )
        if t == len(Dlist) - 1:
            np.savez(
                savedir + "testOutput.npz",
                S=S,
                nu_S=nu_S,
                T=T,
                nu_T=nu_T,
                D=D,
                nu_D=nu_D,
            )
            pointList[int(t * len(D)) : int((t + 1) * len(D))] = D
            pointListG[int(t * len(G)) : int((t + 1) * len(G))] = G
            featList[int(t * len(D)) : int((t + 1) * len(D))] = np.squeeze(
                np.sum(nu_D, axis=-1)
            )[..., None]
            featListG[int(t * len(G)) : int((t + 1) * len(G))] = np.squeeze(nu_G)[
                ..., None
            ]
        else:
            pointList[int(t * len(D)) : int((t + 1) * len(D))] = D
            pointListG[int(t * len(G)) : int((t + 1) * len(G))] = G
            featList[int(t * len(D)) : int((t + 1) * len(D))] = np.squeeze(
                np.sum(nu_D, axis=-1)
            )[..., None]
            featListG[int(t * len(G)) : int((t + 1) * len(G))] = np.squeeze(nu_G)[
                ..., None
            ]
            polyList[int(t * len(D)) : int((t + 1) * len(D)), 1] = np.arange(
                t * len(D), (t + 1) * len(D)
            )
            polyList[int(t * len(D)) : int((t + 1) * len(D)), 2] = np.arange(
                (t + 1) * len(D), (t + 2) * len(D)
            )
            polyListG[int(t * len(G)) : int((t + 1) * len(G)), 1] = np.arange(
                t * len(G), (t + 1) * len(G)
            )
            polyListG[int(t * len(G)) : int((t + 1) * len(G)), 2] = np.arange(
                (t + 1) * len(G), (t + 2) * len(G)
            )

    vtf.writeVTK(
        pointList,
        [featList],
        ["Weights"],
        savedir + "testOutput_curves.vtk",
        polyData=polyList,
    )
    vtf.writeVTK(
        pointListG,
        [featListG],
        ["Weights"],
        savedir + "testOutput_grid.vtk",
        polyData=polyListG,
    )
    volS = np.prod(np.max(S, axis=(0, 1)) - np.min(S, axis=(0, 1)))
    volT = np.prod(np.max(T, axis=(0, 1)) - np.min(T, axis=(0, 1)))
    volD = np.prod(np.max(Dlist[-1], axis=(0, 1)) - np.min(Dlist[-1], axis=(0, 1)))
    getLocalDensity(S, nu_S, sigmaVar[0], savedir + "density_S.vtk", coef=2.0)
    getLocalDensity(T, nu_T, sigmaVar[0], savedir + "density_T.vtk", coef=2.0)
    getLocalDensity(
        Dlist[-1], nu_Dlist[-1], sigmaVar[0], savedir + "density_D.vtk", coef=2.0
    )

    print("volumes of source, target, and deformed source")
    print(volS)
    print(volT)
    print(volD)
    print("total mass")
    wS = np.sum(nu_S)
    wT = np.sum(nu_T)
    wD = np.sum(nu_Dlist[-1])
    print(wS)
    print(wT)
    print(wD)
    print("total mass per total volume")
    print(wS / volS)
    print(wT / volT)
    print(wD / volD)

    optimalParams = np.load(savedir + "testOutput_values.npz")
    A0 = optimalParams["A0"]
    q0 = optimalParams["q0"]
    tau0 = optimalParams["tau0"]

    qx0 = np.reshape(q0[N:], (N, d))
    qw0 = np.reshape(q0[:N], (N, 1))

    x = applyAandTau(qx0, qw0, A0, tau0)
    vtf.writeVTK(
        x, [qw0], ["weights"], savedir + "testOutput_D_ATau.vtk", polyData=None
    )
    getLocalDensity(x, nu_S, sigmaVar[0], savedir + "density_D_ATau.vtk", coef=2.0)

    sys.stdout = original
    return


if __name__ == "__main__":
    main()
