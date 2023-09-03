import numpy as np
import scipy as sp
import torch
from pykeops.torch import Vi,Vj

from xmodmap.io.getOutput import writeParticleVTK

def applyAffine(Z, nu_Z, A, tau, bc=False):
    """
    Makes a new set of particles based on an input set and applying the affine transformation given by matrix A and translation, tau
    """
    print("max before ", torch.max(Z, axis=0))
    R = torch.clone(Z)
    nu_R = torch.clone(nu_Z)
    if not bc:
        R = R @ A.T + tau
    else:
        # rotate around center of mass
        xc = torch.sum((torch.sum(nu_R, axis=-1) * Z), axis=0) / torch.sum(nu_R)
        Rp = R - xc
        R = Rp @ A.T + tau
    print("max ", torch.max(R, axis=0))
    return R, nu_R


def flip(Z):
    R = torch.clone(Z)
    R[:, 0] = -1.0 * R[:, 0]

    return R


def alignBaryCenters(S, nu_S, T, nu_T):
    """
    Translate S to be on barycenter of T with barycenters computed based on sum over features in nu_S and nu_T
    """
    wS = nu_S.sum(axis=-1)
    wT = nu_T.sum(axis=-1)

    xcS = (S * wS).sum(dim=0) / (wS.sum(dim=0))
    xcT = (T * wT).sum(dim=0) / (wT.sum(dim=0))

    tau = xcT - xcS
    Snew = S + tau

    return Snew


def get3DRotMatrix(thetaX, thetaY, thetaZ):
    """
    thetaX, thetaY, and thetaZ should all be torch tensors in radians
    """
    Ax = torch.zeros((3, 3))
    Ax[0, 0] = torch.tensor(1.0)
    Ax[1, 1] = torch.cos(thetaX)
    Ax[2, 2] = Ax[1, 1]
    Ax[1, 2] = -torch.sin(thetaX)
    Ax[2, 1] = torch.sin(thetaX)

    Ay = torch.zeros((3, 3))
    Ay[0, 0] = torch.cos(thetaY)
    Ay[0, 2] = torch.sin(thetaY)
    Ay[2, 0] = -torch.sin(thetaY)
    Ay[2, 2] = torch.cos(thetaY)
    Ay[1, 1] = torch.tensor(1.0)

    Az = torch.zeros((3, 3))
    Az[0, 0] = torch.cos(thetaZ)
    Az[0, 1] = -torch.sin(thetaZ)
    Az[1, 0] = torch.sin(thetaZ)
    Az[1, 1] = torch.cos(thetaZ)
    Az[2, 2] = torch.tensor(1.0)

    R = Ax @ Ay @ Az
    print("R: ", R)
    return R


def get3DRotMatrixAxis(theta, ax=None):
    # ax = vector on unit sphere (e.g.
    # theta is numpy number in radians
    if ax is None:
        ax = np.zeros((3, 1))
        ax[1] = 1.0
    K = np.zeros((3, 3))
    K[0, 1] = -ax[2]
    K[0, 2] = ax[1]
    K[1, 0] = ax[2]
    K[2, 0] = -ax[1]
    K[1, 2] = -ax[0]
    K[2, 1] = ax[0]
    # A = np.cross(-theta*ax,np.identity(3),axisa=0,axisb=0)
    R = sp.linalg.expm(theta * K)
    return torch.from_numpy(R)


def rescaleData(S, T):
    """
    Rescales Data to be in bounding box of [0,1]^d
    Returns rescaled data and rescaling coefficient (one coefficient for each direction)
    """

    X = torch.cat((S, T))
    m = torch.min(X, axis=0).values
    rang = torch.max(X, axis=0).values
    print("min and max originally")
    print(m.detach())
    print(rang.detach())
    rang = rang - m
    s = torch.max(rang)
    Stilde = (S - m) * (1.0 / s)
    Ttilde = (T - m) * (1.0 / s)

    return Stilde, Ttilde, s, m


def rescaleDataList(Slist):
    X = torch.cat((Slist[0], Slist[1]))
    for i in range(2, len(Slist)):
        X = torch.cat((X, Slist[i]))
    m = torch.min(X, axis=0).values
    rang = torch.max(X, axis=0).values - m
    s = torch.max(rang)
    Stilde = []
    for i in range(len(Slist)):
        Stilde.append((Slist[i] - m) * (1.0 / s))
    return Stilde, s, m


def resizeData(Xtilde, s, m):
    """
    Inverse of rescaleData. Takes coefficient of scaling s and rescales it appropriately.
    """
    X = (Xtilde * s) + m
    return X

def rescaleDataSM(X, s, m):
    '''
    Rescales data with given s, m found from rescaling based on S, T
    '''
    Xtilde = (X - m) * (1.0 / s)
    return Xtilde


def scaleDataByVolumes(S, nuS, T, nuT, dRel=3):
    """
    Scale the source by the ratio of the overall volumes. Assume isotropic scaling.
    """
    # center source and target at 0,0
    minS = torch.min(S, axis=0).values
    maxS = torch.max(S, axis=0).values
    print("minS size", minS.shape)
    Sn = S - torch.tensor(0.5) * (maxS + minS)
    if dRel < 3:
        vS = torch.prod(maxS[:dRel] - minS[:dRel])
        vT = torch.prod(
            torch.max(T, axis=0).values[:dRel] - torch.min(T, axis=0).values[:dRel]
        )
        print("vT versus vS")
        print(vT)
        print(vS)
        scaleF = vT / vS
        scaleF = scaleF ** (1.0 / dRel)
        Sn[:, 0:dRel] = Sn[:, 0:dRel] * scaleF
        nuSn = nuS * (scaleF**dRel)
    else:
        vS = torch.prod(maxS - minS)
        vT = torch.prod(torch.max(T, axis=0).values - torch.min(T, axis=0).values)
        scaleF = vT / vS
        scaleF = scaleF ** (1.0 / 3.0)
        Sn = scaleF * Sn
        nuSn = nuS * (scaleF**3)
    print("scale factor is, ", scaleF)

    return Sn, nuSn


def combineFeatures(S, nuS, listOfCols):
    """
    Combine list of nuS columns (feature dimensions) as set of new features
    Remove particles in S that do not have any more mass
    """

    numFeats = len(listOfCols)
    nuSnew = torch.zeros((nuS.shape[0], numFeats))

    c = 0
    for l in listOfCols:
        if len(l) > 1:
            nuSnew[:, c] = torch.sum(nuS[:, l], axis=-1)
        else:
            nuSnew[:, c] = torch.squeeze(nuS[:, l])
        c = c + 1
    toKeep = torch.sum(nuSnew, axis=-1) > 0
    Snew = S[toKeep, ...]
    nuSnew = nuSnew[toKeep, ...]

    return Snew, nuSnew


def checkZ(S, T):
    """
    Make narrowest dimension the Z dimension (e.g. last dimension) to allow for independent scaling.
    """
    r = torch.max(S, axis=0).values - torch.min(S, axis=0).values
    dSmall = np.argmin(r.cpu().numpy())

    if dSmall != 2:
        print("original smallest dimension is ", dSmall)
        Sn = torch.clone(S.detach())
        Sn[:, dSmall] = S[:, 2]
        Sn[:, 2] = S[:, dSmall]
        r = torch.max(Sn, axis=0).values - torch.min(Sn, axis=0).values
        dSmall = torch.argmin(r)
        print("new smallest dimension is ", dSmall)
    else:
        Sn = S

    r = torch.max(T, axis=0).values - torch.min(T, axis=0).values
    dSmall = np.argmin(r.cpu().numpy())

    if dSmall != 2:
        print("original smallest dimension is ", dSmall)
        Tn = torch.clone(T.detach())
        Tn[:, dSmall] = T[:, 2]
        Tn[:, 2] = T[:, dSmall]
        r = torch.max(Tn, axis=0).values - torch.min(Tn, axis=0).values
        dSmall = torch.argmin(r)
        print("new smallest dimension is ", dSmall)
    else:
        Tn = T

    return Sn, Tn

def findSymmetryPlane(T,nu_T,savename,axis=0,angleW=0.0):
    '''
    Find symmetry plane in data by aligning flipped source to original.
    
    Returns rotation and translation needed to align flipped source to original.
    '''
    
    S = torch.clone(T)
    S[:,axis] = -1.0*S[:,axis]
    writeParticleVTK(S,nu_T,savename+"originalflip.vtk",norm=True, condense=True, featNames=None, sW=None)
    
    def getR(angle):
        A = torch.eye(3)
        A[0,0] = torch.cos(angle)
        A[1,1] = torch.cos(angle)
        A[0,1] = -torch.sin(angle)
        A[1,0] = torch.sin(angle)
        return A
    
    def makeLoss(T,nu_T,sigma):
        Tj = Vj(T)/sigma
        sW = (T[:,axis] >= -3.5)*(T[:,axis] <= -1.5) # bounds of comparison (support weights)
        if nu_T.shape[-1] < 60:
            W = (Vi(nu_T)*Vj(sW[...,None]*nu_T)).sum()
        else:
            ssW = sW[...,None]*nu_T
            W = (Vi(nu_T[:,0:10])*Vj(ssW[:,0:10])).sum()
            for j in range(10,nu_T.shape[-1]-10,10):
                W += (Vi(nu_T[:,j:j+10])*Vj(ssW[:,j:j+10])).sum()
                    
        def loss(angle,tau2):
            A = getR(angle)
            tau3 = torch.zeros((1,3))
            tau3[0,0:2] = tau2
            
            dS = S @ A.T + tau3
            Si = Vi(dS)/sigma
            D2 = Si.sqdist(Tj)
            K = (-D2 * 0.5).exp()*W
            return K.sum_reduction(axis=1)
        
        return loss
    
    angle0 = torch.zeros((1)).requires_grad_(True)
    tau0 = torch.zeros((1,2)).requires_grad_(True)
    
    optimizer = torch.optim.LBFGS(
        [angle0,tau0],
        max_eval=15,
        max_iter=10,
        line_search_fn="strong_wolfe",
        history_size=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )
    
    loss = makeLoss(T,nu_T,sigma=1.0)
    
    def closure():
        optimizer.zero_grad()
        L = loss(angle0,tau0)
        L = -L.sum() + angleW*angle0**2 # limit rotation 
        print("loss: ", L.detach())
        L.backward()
        return L
    
    for i in range(10):
        optimizer.step(closure)
    
    A=getR(angle0.detach())
    tau3 = torch.zeros((1,3))
    tau3[0,0:2] = tau0.detach()
    
    dS = S @ A.T + tau3
    
    params = {
        "A": A,
        "angle": angle0,
        "tau": tau0,
        "D": dS
    }
    
    torch.save(params,savename+'.pt')
    print("A: ", A)
    print("angle: ", angle0)
    print("tau: ", tau0)
    return A,tau3,dS

def makeWhole(T,nu_T,savename,axis=0,thresh=0.02):
    '''
    Make whole slice by mapping reflection onto original slice.
    Take union of points (remove those too close to originals).
    
    if minimum distance is less than thresh = remove points
    '''
    A,tau3,dS = findSymmetryPlane(T,nu_T,savename,axis)
    
    # save deformed reflection
    writeParticleVTK(dS, nu_T, savename + 'dS.vtk', norm=True, condense=True, featNames=None, sW=None)
    writeParticleVTK(T,nu_T,savename+'.vtk', norm=True, condense=True, featNames=None, sW=None)
    
    Di = Vi(dS)
    Tj = Vj(T)
    
    D2 = Di.sqdist(Tj)
    minDist = D2.min(axis=1) # find minimum distance to point
    indsToKeep = torch.squeeze(minDist > thresh)
    D = dS[indsToKeep,:]
    nu_D = nu_T[indsToKeep,:]
    
    S = torch.cat((T,D))
    nu_S = torch.cat((nu_T,nu_D))
    
    writeParticleVTK(S,nu_S,savename+'total.vtk',norm=True,condense=True,featNames=None,sW=None)
    np.savez(savename+'total.npz',X=S.cpu().numpy(),nu_X=nu_S.cpu().numpy())
    return




    
    
