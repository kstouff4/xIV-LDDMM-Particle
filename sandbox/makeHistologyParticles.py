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
    outpath = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/Exvivohuman_11T/Brain3/"

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    outpath = outpath + "Block3_down_016_t5of15_brown/"

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    imgfils = [
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L1_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L5_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L10_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L15_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L20_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L25_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L30_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L35_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L40_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L45_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L50_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L55_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip1/BR_3_Bl_1_L60_3.TIF",
    ]

    imgfils = [
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_1_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_5_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_10_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_15_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_20_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_25_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_30_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_35_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_40_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_45_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_50_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_55_3.TIF",
        "/cis/home/kstouff4/Documents/datasets/exvivohuman_11T/more_blocks/Brain3/histology/Restain_2023/AD_Hip3/BR1_BL3_L_60_3.TIF",
    ]

    res = [0.00208, 0.00208, 0.00208]
    res = [0.0083, 0.0083, 0.0083]
    res = [0.0333, 0.0333, 0.01]
    for i in range(len(imgfils)):
        imgfile = imgfils[i]
        # S,nu_S = gI.makeBinsFromMultiChannelImage(imgfile,res,2,3,ds=16,z=(i+1.0),threshold=2,bins=10,reverse=True)
        S, nu_S = gI.makeBinsFromDistance(
            imgfile,
            res,
            2,
            [155, 110, 100],
            ds=16,
            z=(i + 1.0),
            threshold=10,
            bins=20,
            weights=[1, 1, 1],
        )
        # S,nu_S = gI.makeFromSingleChannelImage(imgfile,res[0],bg=[0],ordering=[1,2,3],ds=1)

        S = S.detach().cpu().numpy()
        nu_S = nu_S.detach().cpu().numpy()

        fname = outpath + imgfile.split("/")[-1].replace(".img", ".npz").replace(
            ".nii.gz", ".npz"
        ).replace(".tif", ".npz").replace(".TIF", ".npz")
        np.savez(fname, S=S, nu_S=nu_S)
        imagenames = ["TotalMass", "MaxBin"]
        imageVals = [np.sum(nu_S, axis=-1), np.argmax(nu_S, axis=-1)]
        for i in range(nu_S.shape[-1]):
            imagenames.append("Bin_" + str(i))
            imageVals.append(nu_S[:, i])
        vtf.writeVTK(
            S, imageVals, imagenames, fname.replace(".npz", ".vtk"), polyData=None
        )

    return


if __name__ == "__main__":
    main()
