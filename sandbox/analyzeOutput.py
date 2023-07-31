import numpy as np
from matplotlib import pyplot as plt
from sys import path as sys_path

sys_path.append("../")

import xmodmap.io.getOutput as gO

import scipy as sp
from scipy import linalg

import nibabel as nib
from scipy.ndimage import gaussian_filter

import torch
from pykeops.torch import Vi, Vj

np_dtype = "float32"  # "float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor  # DoubleTensor
else:
    dtype = torch.FloatTensor

import xmodmap


def getLocalDensity(Zi, nu_Zi, sigma, savename, coef=3):
    """
    Compute local density in cube of size 2sigma x 2sigma x 2sigma
    """
    if torch.is_tensor(Zi):
        Z = Zi.cpu().numpy()
        nu_Z = nu_Zi.cpu().numpy()
    else:
        Z = Zi
        nu_Z = nu_Zi
    if len(nu_Z.shape) < 2:
        nu_Z = nu_Z[..., None]
    if nu_Z.shape[-1] == 1:
        nu_Z = np.zeros_like(nu_Z) + 1.0
        print("nu_Z shape is, ", nu_Z.shape)
    cSize = coef * sigma
    bbSize = np.round(1 + np.max(Z, axis=(0, 1)) - np.min(Z, axis=(0, 1)) - 1)

    coords_labels = np.floor((Z - np.floor(np.min(Z, axis=0))) / cSize).astype(
        int
    )  # minimum number of cubes in x and y
    totCubes = (
        (np.max(coords_labels[:, 0]) + 1)
        * (np.max(coords_labels[:, 1]) + 1)
        * (np.max(coords_labels[:, 2]) + 1)
    )
    xC = (
        np.arange(np.max(coords_labels[:, 0]) + 1) * cSize
        + np.floor(np.min(Z[:, 0]))
        + cSize / 2.0
    )
    yC = (
        np.arange(np.max(coords_labels[:, 1]) + 1) * cSize
        + np.floor(np.min(Z[:, 1]))
        + cSize / 2.0
    )
    zC = (
        np.arange(np.max(coords_labels[:, 2]) + 1) * cSize
        + np.floor(np.min(Z[:, 2]))
        + cSize / 2.0
    )

    XC, YC, ZC = np.meshgrid(xC, yC, zC, indexing="ij")
    cubes_centroids = np.stack((XC, YC, ZC), axis=-1)
    cubes_indices = np.reshape(
        np.arange(totCubes),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    coords_labels_tot = cubes_indices[
        coords_labels[:, 0], coords_labels[:, 1], coords_labels[:, 2]
    ]

    cubes_mrna = np.zeros((totCubes, nu_Z.shape[-1]))
    for c in range(totCubes):
        cubes_mrna[c, :] = np.sum(nu_Z[coords_labels_tot == c, :], axis=0)
    centroidsPlot = np.zeros((totCubes, 3))
    centroidsPlot[:, 0] = np.ravel(XC)
    centroidsPlot[:, 1] = np.ravel(YC)
    centroidsPlot[:, 2] = np.ravel(ZC)

    cubes_mrna = cubes_mrna / (cSize**3)
    imageNames = []
    imageDensity = []
    imageNames.append("TotalDensity")
    imageDensity.append(np.sum(cubes_mrna, axis=-1))
    cubes_mrna_prob = xmodmap.normalize(cubes_mrna)
    if nu_Z.shape[-1] > 1:
        for f in range(nu_Z.shape[-1]):
            imageNames.append("Feature" + str(f) + "_Density")
            imageDensity.append(cubes_mrna[:, f])
        for f in range(nu_Z.shape[-1]):
            imageNames.append(f"Feature{f}_Probability")
            imageDensity.append(cubes_mrna_prob[:, f])
    gO.writeVTK(centroidsPlot, imageDensity, imageNames, savename, polyData=None)
    return


def getCompareDensity(T, nu_T, D, nu_D, sigma, savedir, coef=3):
    """
    Compute local density in cube of size 2sigma x 2sigma x 2sigma; save as segmentation image (maxVal) and intensity (weights)
    """
    cSize = coef * sigma
    miT = np.min(T, axis=0)
    maT = np.max(T, axis=0)
    miD = np.min(D, axis=0)
    maD = np.max(D, axis=0)

    mi = np.vstack((miT, miD))
    ma = np.vstack((maT, maD))
    mi = np.min(mi, axis=0)
    ma = np.max(ma, axis=0)
    bbSize = np.round((1 + ma) - (mi - 1))

    coords_labels = np.floor((T - np.floor(np.min(T, axis=0))) / cSize).astype(
        int
    )  # minimum number of cubes in x and y
    coords_labels = np.ceil((np.ceil(ma) - np.floor(mi)) / cSize).astype(
        int
    )  # number of cubes in x, y, and z
    print(coords_labels.shape)
    totCubes = (
        (np.max(coords_labels[0]))
        * (np.max(coords_labels[1]))
        * (np.max(coords_labels[2]))
    )
    xC = np.arange(np.max(coords_labels[0])) * cSize + np.floor(mi[0]) + cSize / 2.0
    yC = np.arange(np.max(coords_labels[1])) * cSize + np.floor(mi[1]) + cSize / 2.0
    zC = (
        np.arange(np.max(coords_labels[2])) * cSize + np.floor(mi[2]) + cSize / 2.0
    )  # center in middle of cube

    XC, YC, ZC = np.meshgrid(xC, yC, zC, indexing="ij")  # physical coordinates
    cubes_centroids = np.stack((XC, YC, ZC), axis=-1)
    print("cubes_centroids shapes, ", cubes_centroids.shape)
    cubes_indices = np.reshape(
        np.arange(totCubes),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    # coords_labels_tot = cubes_indices[coords_labels[:,0],coords_labels[:,1],coords_labels[:,2]]

    # assign each measure to 1 cube
    coords_labelsT = np.floor((T - np.floor(mi)) / cSize).astype(int)
    coords_labelsD = np.floor((D - np.floor(mi)) / cSize).astype(int)

    coords_labelsTtot = cubes_indices[
        coords_labelsT[:, 0], coords_labelsT[:, 1], coords_labelsT[:, 2]
    ]
    coords_labelsDtot = cubes_indices[
        coords_labelsD[:, 0], coords_labelsD[:, 1], coords_labelsD[:, 2]
    ]

    cubes_nuT = np.zeros((totCubes, nu_T.shape[-1]))
    cubes_nuD = np.zeros((totCubes, nu_D.shape[-1]))

    for c in range(totCubes):
        cubes_nuT[c, :] = np.sum(nu_T[coords_labelsTtot == c, :], axis=0)
        cubes_nuD[c, :] = np.sum(nu_D[coords_labelsDtot == c, :], axis=0)

    # save densities as images
    densT = np.reshape(
        np.sum(cubes_nuT, axis=-1),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    densD = np.reshape(
        np.sum(cubes_nuD, axis=-1),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    empty_header = nib.Nifti1Header()
    densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
    densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
    densTDim = nib.Nifti1Image(np.sqrt((densT - densD) ** 2), np.eye(4), empty_header)

    nib.save(densTim, savedir + "Tdensity.nii.gz")
    nib.save(densDim, savedir + "Ddensity.nii.gz")
    nib.save(densTDim, savedir + "TDdiffdensity.nii.gz")

    densT = np.reshape(
        (np.sum(cubes_nuT, axis=-1) > 0) * (np.argmax(cubes_nuT, axis=-1) + 1),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    densD = np.reshape(
        (np.sum(cubes_nuD, axis=-1) > 0) * (np.argmax(cubes_nuD, axis=-1) + 1),
        (cubes_centroids.shape[0], cubes_centroids.shape[1], cubes_centroids.shape[2]),
    )
    empty_header = nib.Nifti1Header()
    densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
    densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
    densTDim = nib.Nifti1Image(
        (densT - densD != 0).astype(int), np.eye(4), empty_header
    )

    nib.save(densTim, savedir + "Tmaxval.nii.gz")
    nib.save(densDim, savedir + "Dmaxval.nii.gz")
    nib.save(densTDim, savedir + "TDdiffmaxval.nii.gz")

    for i in range(nu_T.shape[-1]):
        densT = np.reshape(
            cubes_nuT[..., i],
            (
                cubes_centroids.shape[0],
                cubes_centroids.shape[1],
                cubes_centroids.shape[2],
            ),
        )
        densD = np.reshape(
            cubes_nuD[..., i],
            (
                cubes_centroids.shape[0],
                cubes_centroids.shape[1],
                cubes_centroids.shape[2],
            ),
        )
        empty_header = nib.Nifti1Header()
        densTim = nib.Nifti1Image(densT, np.eye(4), empty_header)
        densDim = nib.Nifti1Image(densD, np.eye(4), empty_header)
        densTDim = nib.Nifti1Image(
            (densT - densD != 0).astype(int), np.eye(4), empty_header
        )

        nib.save(densTim, savedir + "Tnu_" + str(i) + ".nii.gz")
        nib.save(densDim, savedir + "Dnu_" + str(i) + ".nii.gz")
        nib.save(densTDim, savedir + "TDdiffnu_" + str(i) + ".nii.gz")

    np.savez(
        savedir + "TD_values.npz",
        cubes_nuT=cubes_nuT,
        cubes_nuD=cubes_nuD,
        cubes_centroids=cubes_centroids,
        XC=XC,
        YC=YC,
        ZC=ZC,
        cSize=cSize,
    )

    """
    centroidsPlot = np.zeros((totCubes,3))
    centroidsPlot[:,0] = np.ravel(XC)
    centroidsPlot[:,1] = np.ravel(YC)
    centroidsPlot[:,2] = np.ravel(ZC)
    
    cubes_mrna = cubes_mrna/(cSize**3)
    imageNames = []
    imageDensity = []
    imageNames.append('TotalDensity')
    imageDensity.append(np.sum(cubes_mrna,axis=-1))
    if (nu_Z.shape[-1] > 1):
        for f in range(nu_Z.shape[-1]):
            imageNames.append('Feature' + str(f) + '_Density')
            imageDensity.append(cubes_mrna[:,f])
    vtf.writeVTK(centroidsPlot,imageDensity,imageNames,savename,polyData=None)
    """
    return


def applyAandTau(q_x, q_w, A, tau):
    """
    q_x indicates the original positions of the source and q_w the original weights
    arguments are numpy arrays
    """
    x_c0 = np.sum(q_w * q_x, axis=0) / np.sum(q_w)
    x = (q_x - x_c0) @ ((sp.linalg.expm(A)).T) + tau

    return x


def smootheIm(npz, sig=1.0):
    n = np.load(npz)
    X = n["cubes_nuD"]
    cubes_centroids = n["cubes_centroids"]

    Xd = np.reshape(
        X,
        (
            cubes_centroids.shape[0],
            cubes_centroids.shape[1],
            cubes_centroids.shape[2],
            X.shape[-1],
        ),
    )
    Xdnew = np.zeros_like(Xd)

    for f in range(Xd.shape[-1]):
        Xdnew[..., f] = gaussian_filter(Xd[..., f], sig)
        print("mass before is: ", np.sum(Xd[..., f]))
        print("mass after is: ", np.sum(Xdnew[..., f]))

    np.savez(npz.replace(".npz", "_smoothed_sig" + str(sig) + ".npz"), cubes_nuD=Xdnew)
    empty_header = nib.Nifti1Header()
    densim = nib.Nifti1Image(np.sum(Xdnew, axis=-1), np.eye(4), empty_header)
    maxValim = nib.Nifti1Image(np.argmax(Xdnew, axis=-1), np.eye(4), empty_header)

    nib.save(
        densim, npz.replace(".npz", "_smoothed_sig" + str(sig) + "Ddensity.nii.gz")
    )
    nib.save(
        maxValim, npz.replace(".npz", "_smoothed_sig" + str(sig) + "DmaxVal.nii.gz")
    )
    return


def interpolateNN(Td, T, S, nu_S, pi_ST, savename):
    """
    interpolate high resolution source (S)
    npzT = npz with deformed target
    pi_ST = transfer matrix for weights
    """

    T_i = Vi(torch.tensor(Td).type(dtype))
    S_j = Vj(torch.tensor(S).type(dtype))

    D_ij = ((T_i - S_j) ** 2).sum(-1)  # symbolic matrix of squared distances
    indKNN = D_ij.argKmin(1, dim=1)  # get nearest neighbor for each of target points
    print("indKNN shape, ", indKNN.shape)
    print("nu_S shape, ", nu_S.shape)

    nu_TS = nu_S[
        np.squeeze(indKNN.cpu().numpy()), ...
    ]  # Assign target points with feature values of source
    print("nu_TS shape, ", nu_TS.shape)
    nu_TSpi = nu_TS @ pi_ST
    print("nu_TSpi shape, ", nu_TSpi.shape)

    np.savez(savename, nu_TS=nu_TS, nu_TSpi=nu_TSpi)
    imageVals = [np.sum(nu_TSpi, axis=-1), np.argmax(nu_TSpi, axis=-1)]
    imageNames = ["TotalMass", "MaxVal"]
    zeta_TSpi = nu_TSpi / np.sum(nu_TSpi, axis=-1)[..., None]
    print("zeta_TSpi shape, ", zeta_TSpi.shape)
    for f in range(nu_TSpi.shape[-1]):
        imageVals.append(zeta_TSpi[:, f])
        imageNames.append("Zeta_" + str(f))

    gO.writeVTK(
        T, imageVals, imageNames, savename.replace(".npz", ".vtk"), polyData=None
    )
    return T, nu_TSpi


def removeZerosAndNormalize(imgT, imgD, norm=True):
    iT = nib.load(imgT)
    iTim = np.asarray(iT.dataobj)
    iD = nib.load(imgD)
    iDim = np.asarray(iD.dataobj)

    print("unique values")
    print(np.unique(iTim))
    print(np.unique(iDim))

    zNonzero = []
    for z in range(iDim.shape[2]):
        if np.sum(iTim[:, :, z]) > 0:
            zNonzero.append(z)
    iTimNZ = iTim[:, :, zNonzero]
    iDimNZ = iDim[:, :, zNonzero]
    if norm:
        iTimNZ = iTimNZ / np.max(iTimNZ)
        iDimNZ = iDimNZ / np.max(iDimNZ)

    empty_header = nib.Nifti1Header()
    iTimSave = nib.Nifti1Image(iTimNZ, np.eye(4), empty_header)
    iDimSave = nib.Nifti1Image(iDimNZ, np.eye(4), empty_header)

    nib.save(iTimSave, imgT.replace(".nii.gz", "_RZN.nii.gz"))
    nib.save(iDimSave, imgD.replace(".nii.gz", "_RZN.nii.gz"))
    return


def interpolateWithImage(imgS, res, Tdgrid, Tgrid, savename, flip=False, ds=2):
    """
    Tdgrid should be a regular grid covering the support of the target slice, deformed into atlas space
    Tgrid should be in X x Y x Z x 3 shape
    """
    iS = nib.load(imgS)
    iSim = np.asarray(iS.dataobj)
    print("shape of atlas: ", iSim.shape)
    iSim = iSim[0::ds, 0::ds, 0::ds, ...]
    print("ranges of image")
    print(iSim.shape)
    print(np.max(iSim))
    print(np.min(iSim))

    x0 = np.arange(iSim.shape[0]) * res[0] * ds
    x1 = np.arange(iSim.shape[1]) * res[1] * ds
    x2 = np.arange(iSim.shape[2]) * res[2] * ds

    x0 = x0 - np.mean(x0)
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    if flip:
        x2 = -1.0 * x2

    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
    nuX = np.ravel(iSim)
    X0r = np.ravel(X0)
    X1r = np.ravel(X1)
    X2r = np.ravel(X2)
    X = np.zeros((X0r.shape[0], 3))
    X[:, 0] = X0r
    X[:, 1] = X1r
    X[:, 2] = X2r
    print("X shape, ", X.shape)
    print(np.min(X, axis=0))
    print(np.max(X, axis=0))
    print(np.min(Tdgrid, axis=0))
    print(np.max(Tdgrid, axis=0))

    Tgridr = np.reshape(Tgrid, (Tgrid.shape[0] * Tgrid.shape[1] * Tgrid.shape[2], 3))
    mi = np.min(Tgridr, axis=0) - np.asarray(res) * 2.0
    ma = np.max(Tgridr, axis=0) + np.asarray(res) * 2.0

    inds = (
        (X[:, 0] > mi[0])
        * (X[:, 0] < ma[0])
        * (X[:, 1] > mi[1])
        * (X[:, 1] < ma[1])
        * (X[:, 2] > mi[2])
        * (X[:, 2] < ma[2])
    )
    Xnew = X[inds, ...]
    nuXnew = nuX[inds, ...]
    print("X new shape, ", Xnew.shape)
    print(np.min(Xnew))
    print(np.max(Xnew))

    T_i = Vi(torch.tensor(Tdgrid).type(dtype))
    S_j = Vj(torch.tensor(Xnew).type(dtype))

    D_ij = ((T_i - S_j) ** 2).sum(-1)  # symbolic matrix of squared distances
    indKNN = D_ij.argKmin(1, dim=1)  # get nearest neighbor for each of target points

    nu_TS = nuXnew[
        np.squeeze(indKNN.cpu().numpy()), ...
    ]  # Assign target points with feature values of source
    print("nu_TS shape, ", nu_TS.shape)
    print(np.unique(nu_TS))

    nu_TSim = np.reshape(nu_TS, (Tgrid.shape[0], Tgrid.shape[1], Tgrid.shape[2]))
    print("nu_TS shape image, ", nu_TSim.shape)
    np.savez(savename + ".npz", nu_TS=nu_TS, nu_TSim=nu_TSim)

    if Tgrid.shape[2] < 2:
        f, ax = plt.subplots()
        ax.imshow(nu_TSim, cmap="gray")
        f.savefig(savename + ".png", dpi=300)

    empty_header = nib.Nifti1Header()
    wIm = nib.Nifti1Image(nu_TSim[..., None], np.eye(4), empty_header)
    nib.save(wIm, savename + ".nii.gz")

    return
