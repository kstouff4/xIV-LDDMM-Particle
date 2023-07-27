import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path

sys_path.append("..")
sys_path.append("../xmodmap")
sys_path.append("../xmodmap/io")
import initialize as init
import getInput as gI

sys_path.append("/cis/home/kstouff4/Documents/SurfaceTools/")
import vtkFunctions as vtf

import torch

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
np_dtype = "float32"  # "float64"
dtype = torch.cuda.FloatTensor  # DoubleTensor

import nibabel as nib
import alignMRIBlocks as amb

##########################################################################


def main():
    sigmaVar = [50.0, 20.0, 10.0]
    gammaR = 1.0
    gammaC = 1.0
    savedir = "/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/Human/Exvivohuman_11T/"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = savedir + "JOG57/"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = savedir + "Params" + str(sigmaVar) + str(gammaR) + str(gammaC) + "/"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    zCoordsO = [
        torch.tensor(149.0).type(dtype),
        torch.tensor(0.0).type(dtype),
        torch.tensor(-140.0).type(dtype),
    ]

    b1S, b1nu_S = gI.getFromFile(
        "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57_1_ap.nii._rotated.npz"
    )
    b2S, b2nu_S = gI.getFromFile(
        "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57_2_ap.nii._rotated.npz"
    )
    b3S, b3nu_S = gI.getFromFile(
        "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57_3_ap.nii._rotated.npz"
    )
    b4S, b4nu_S = gI.getFromFile(
        "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/JOG57_4_ap.nii._rotated.npz"
    )

    sigmaVarList = []
    for sigg in sigmaVar:
        sigmaVarList.append(torch.tensor(sigg).type(dtype))

    ASlistnp, Asnp, tsnp = amb.callOptimize(
        [b1S, b2S, b3S, b4S],
        [b1nu_S, b2nu_S, b3nu_S, b4nu_S],
        sigmaVarList,
        torch.tensor(gammaR).type(dtype),
        savedir,
        its=50,
        d=3,
        numVars=6,
    )

    bSorig = [
        b1S.detach().cpu().numpy(),
        b2S.detach().cpu().numpy(),
        b3S.detach().cpu().numpy(),
        b4S.detach().cpu().numpy(),
    ]
    bnuSorig = [
        b1nu_S.detach().cpu().numpy(),
        b2nu_S.detach().cpu().numpy(),
        b3nu_S.detach().cpu().numpy(),
        b4nu_S.detach().cpu().numpy(),
    ]
    imagenames = ["TotalMass", "MaxBin", "Bin_0", "Bin_1", "Bin_2"]

    for i in range(4):
        imageVals = []
        imageVals.append(np.sum(bnuSorig[i], axis=-1))
        imageVals.append(np.argmax(bnuSorig[i], axis=-1))
        imageVals.append(bnuSorig[i][:, 0])
        imageVals.append(bnuSorig[i][:, 1])
        imageVals.append(bnuSorig[i][:, 2])
        vtf.writeVTK(
            ASlistnp[i],
            imageVals,
            imagenames,
            savedir + "transformSeg_" + str(i) + ".vtk",
            polyData=None,
        )
    return


if __name__ == "__main__":
    main()
