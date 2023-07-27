import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path

sys_path.append("..")
sys_path.append("../xmodmap")
sys_path.append("../xmodmap/io")
import getInput as gI

sys_path.append("/cis/home/kstouff4/Documents/SurfaceTools/")
import vtkFunctions as vtf

import torch

np_dtype = "float32"
dtype = torch.cuda.FloatTensor

import nibabel as nib


##################################################################################
def main():
    d = 3
    labs = 13

    outpath = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/FanMERFISH/"

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    fSf = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/gene_S2R2.csv"
    fSs = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/meta_S2R2.csv"

    fTs = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/meta_S2R1.csv"
    fTf = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/gene_S2R1.csv"

    gNames = [
        "Ntrk3",
        "Fzd3",
        "Baiap2",
        "Slc17a6",
        "Adora2a",
        "Gpr151",
        "Gabbr2",
        "Cckar",
        "Adgrb3",
        "Lmtk2",
        "Adgrl1",
        "Cx3cl1",
        "Epha4",
    ]
    S, nu_S = gI.readSpaceFeatureCSV(
        fSs, ["center_x", "center_y"], fSf, gNames, scale=1e-3, labs=labs
    )

    T, nu_T = gI.readSpaceFeatureCSV(
        fTs, ["center_x", "center_y"], fTf, gNames, scale=1e-3, labs=labs
    )

    S = S.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    print("shapes")
    print(S.shape)
    print(nu_S.shape)
    print(T.shape)
    print(nu_T.shape)

    np.savez(outpath + fSf.split("/")[-1].replace("csv", "npz"), X=S, nu_X=nu_S)
    np.savez(outpath + fTf.split("/")[-1].replace("csv", "npz"), X=T, nu_X=nu_T)

    nu_SList = list(nu_S.T)
    nu_SList.append(np.sum(nu_S, axis=-1))
    nu_SList.append(np.argmax(nu_S, axis=-1))
    gNames.append("weight")
    gNames.append("maxVal")

    vtf.writeVTK(S, nu_SList, gNames, fSf.split("/")[-1].replace("csv", "vtk"))
    nu_TList = list(nu_T.T)
    nu_TList.append(np.sum(nu_T, axis=-1))
    nu_TList.append(np.argmax(nu_T, axis=-1))
    vtf.writeVTK(T, nu_TList, gNames, fTf.split("/")[-1].replace("csv", "vtk"))

    fSf = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/cell_S2R2.csv"
    fTf = "/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/cell_S2R1.csv"

    fNames = [
        "Astrocyte1",
        "Astrocyte2",
        "Astrocyte3",
        "Astrocyte4",
        "Astrocyte5",
        "Cortical_Excitatory_Neuron1",
        "Cortical_Excitatory_Neuron2",
        "Endothelial1",
        "Endothelial2",
        "Ependymal",
        "Excitatory_Granule",
        "Excitatory_Neuron1",
        "Excitatory_Neuron2",
        "Excitatory_Pyramidal1",
        "Excitatory_Pyramidal2",
        "GABAergic_Interneuron1",
        "GABAergic_Interneuron2",
        "GABAergic_ER",
        "Inhibitory_Interneuron",
        "Microglia1",
        "Microglia2",
        "OLs_Neurons",
        "OL_Progenitor1",
        "OL_Progenitor2",
        "OL1",
        "OL2",
        "OL3",
        "OL4",
        "OL5",
        "OL6",
        "Pericyte",
    ]
    S, nu_S = gI.readSpaceFeatureCSV(
        fSs, ["center_x", "center_y"], fSf, ["celllabels"], scale=1e-3, labs=None
    )

    T, nu_T = gI.readSpaceFeatureCSV(
        fTs, ["center_x", "center_y"], fTf, ["celllabels"], scale=1e-3, labs=None
    )

    S = S.detach().cpu().numpy()
    nu_S = nu_S.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    nu_T = nu_T.detach().cpu().numpy()

    print("shapes")
    print(S.shape)
    print(nu_S.shape)
    print(T.shape)
    print(nu_T.shape)

    np.savez(outpath + fSf.split("/")[-1].replace("csv", "npz"), X=S, nu_X=nu_S)
    np.savez(outpath + fTf.split("/")[-1].replace("csv", "npz"), X=T, nu_X=nu_T)

    nu_SList = list(nu_S.T)
    nu_SList.append(np.sum(nu_S, axis=-1))
    nu_SList.append(np.argmax(nu_S, axis=-1))
    fNames.append("weight")
    fNames.append("maxVal")

    vtf.writeVTK(S, nu_SList, fNames, fSf.split("/")[-1].replace("csv", "vtk"))
    nu_TList = list(nu_T.T)
    nu_TList.append(np.sum(nu_T, axis=-1))
    nu_TList.append(np.argmax(nu_T, axis=-1))
    vtf.writeVTK(T, nu_TList, fNames, fTf.split("/")[-1].replace("csv", "vtk"))

    return


if __name__ == "__main__":
    main()
