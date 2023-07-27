import sys
from sys import path as sys_path

sys_path.append("../")
sys_path.append("../../")
sys_path.append("../../xmodmap/")
sys_path.append("../../xmodmap/io/")
import shootParticles as sp
import saveState as ss
from analyzeOutput import interpolateWithImage
import initialize as init

import torch

np_dtype = "float32"  # "float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor  # DoubleTensor
else:
    dtype = torch.FloatTensor

import numpy as np


def main():
    indices = [0, 9, 19, 29, 39]  # 0, 9, 19, 29, 39
    resAtlas = [0.01, 0.01, 0.01]
    imgS = "/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap_672.img"
    res = 0.001  # TO DO --> check whether this refers to scale of 1 or in um
    savepref = "/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlas200ToBarSeq/output_dl_sig_its_csgamma_N-35_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_4_10.010000.00.1_67827flipFullAtlasLamb/"
    savepref = "/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlas200ToBarSeq28/output_dl_sig_its_csgamma_N-328_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_100_10.010000.00.1_67827flipFullAtlasLamb/"
    targHigh = "/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeqAligned/top28MI/sig0.1_dimEff2/initialHigh_All.npz"
    # savepref='/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlas200ToBarSeq/output_dl_sig_its_csgamma_N-35_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_2_10.010000.00.1_67827flipFullAtlasLamb/'
    paramsFile = savepref + "State__params.pt"

    # _,_,_,_,s,m = ss.loadVariables(paramsFile.replace('params','variables'))

    # variablesFile = savepref + 'testOutput_values.npz'
    # v = np.load(variablesFile)
    # p0 = v['p0']
    # q0 = v['q0']
    variablesFile = savepref + "State__variables.pt"
    q0, p0, Ttilde, wT, s, m = ss.loadVariables(variablesFile)
    v = np.load(savepref + "testOutput_targetScaled.npz")
    zeta_T = torch.tensor(v["zeta_T"]).type(dtype)
    nu_T = wT * zeta_T

    # get high resolution Target
    if targHigh is not None:
        fils = np.load(targHigh)
        T = fils[fils.files[0]]
        nu_T = fils[fils.files[1]]
        Ttilde = (T - m) * (1.0 / s)
        Ttilde = torch.tensor(Ttilde).type(dtype)
        nu_T = torch.tensor(nu_T).type(dtype)

    # ss.saveVariables(torch.tensor(q0).type(dtype),torch.tensor(p0).type(dtype),Ttilde,wT,s,m,savepref + 'State_')

    for index in indices:
        Xg, nu_Xg, X, nu_X = sp.shootTargetGridSlice(
            Ttilde, nu_T, index, res, savepref, paramsFile, variablesFile
        )  # gives locations and feature values of target
        interpolateWithImage(
            imgS, resAtlas, X, Xg, savepref + "Slice" + str(index) + "_", flip=True
        )
        info = np.load(savepref + "Slice_" + str(index) + "_originalAndShot.npz")
        Xo = init.resizeData(info["X"], s, m)
        nuXo = info["nu_X"]
        Xog = np.reshape(Xo, Xg.shape)
        nuXog = np.reshape(nuXo, nu_Xg.shape)
        interpolateWithImage(
            imgS,
            resAtlas,
            Xo,
            Xog,
            savepref + "Slice" + str(index) + "_startPosition_",
            flip=True,
        )

    return


if __name__ == "__main__":
    main()
