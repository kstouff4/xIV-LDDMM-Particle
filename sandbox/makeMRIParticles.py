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


def main():
    outpath = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/"

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    imgfile = "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/MRI/20161206Hip1_b0.img"
    imgfile = "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/newBrains/JOG57/MRI/JOG57_4_ap.nii._rotated.nii.gz"
    res = [0.1, 0.1, 0.1]
    # res = [1.0,1.0,1.0]
    # S,nu_S = gI.makeBinsFromMultiChannelImage(imgfile,res,3,1,ds=8,z=0,threshold=1,bins=25)
    S, nu_S = gI.makeFromSingleChannelImage(
        imgfile, res[0], bg=[0], ordering=[1, 2, 3], ds=1
    )

    S = S.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()

    fname = outpath + imgfile.split("/")[-1].replace(".img", ".npz").replace(
        ".nii.gz", ".npz"
    )
    np.savez(fname, S=S, nu_S=nu_S)
    imagenames = ["TotalMass", "MaxBin"]
    imageVals = [np.sum(nu_S, axis=-1), np.argmax(nu_S, axis=-1)]
    for i in range(nu_S.shape[-1]):
        imagenames.append("Bin_" + str(i))
        imageVals.append(nu_S[:, i])
    vtf.writeVTK(S, imageVals, imagenames, fname.replace(".npz", ".vtk"), polyData=None)

    return


if __name__ == "__main__":
    main()
