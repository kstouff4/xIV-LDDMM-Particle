import os
import time
import numpy as np
from numpy import random

import torch
from torch.autograd import grad

import pykeops
import socket

pykeops.set_build_folder(
    "~/.cache/keop" + pykeops.__version__ + "_" + (socket.gethostname())
)

from pykeops.torch import Vi, Vj

np_dtype = "float32"
dtype = torch.cuda.FloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from matplotlib import pyplot as plt
import matplotlib

if "DISPLAY" in os.environ:
    matplotlib.use("qt5Agg")
else:
    matplotlib.use("Agg")

import sys
from sys import path as sys_path

sys_path.append("/cis/home/kstouff4/Documents/SurfaceTools/")
import vtkFunctions as vtf


# Kernels
def GaussKernelHamiltonian(sigma, d):
    qx, qy, px, py, wpx, wpy = (
        Vi(0, d) / sigma,
        Vj(1, d) / sigma,
        Vi(2, d),
        Vj(3, d),
        Vi(4, 1) / sigma,
        Vj(5, 1) / sigma,
    )
    h = (
        0.5 * (px * py).sum()
        + wpy * ((qx - qy) * px).sum()
        - (0.5) * wpx * wpy * (qx.sqdist(qy) - d)
    )
    D2 = qx.sqdist(qy)
    K = (-D2 * 0.5).exp()
    return (K * h).sum_reduction(axis=1)  # N x M


def GaussKernelHamiltonianAT(sigma, d):
    qx, qy, px, py, wpx, wpy = (
        Vi(0, d) / sigma,
        Vj(1, d) / sigma,
        Vi(2, d),
        Vj(3, d),
        Vi(4, 1) / sigma,
        Vj(5, 1) / sigma,
    )
    h = (
        0.5 * (px * py).sum()
        + wpy * ((qx - qy) * px).sum()
        - (0.5) * wpx * wpy * (qx.sqdist(qy) - d)
    )
    D2 = qx.sqdist(qy)
    K = (-D2 * 0.5).exp()
    return (K * h).sum_reduction(axis=1)  # N x M


def GaussKernelU(sigma, d):
    x, qy, py, wpy = Vi(0, d) / sigma, Vj(1, d) / sigma, Vj(2, d), Vj(3, 1) / sigma
    D2 = x.sqdist(qy)
    K = (-D2 * 0.5).exp()  # G x N
    h = py + wpy * (x - qy)  # 1 X N x 3
    return (K * h).sum_reduction(axis=1)  # G x 3


def GaussKernelUdiv(sigma, d):
    x, qy, py, wpy = Vi(0, d) / sigma, Vj(1, d) / sigma, Vj(2, d), Vj(3, 1) / sigma
    D2 = x.sqdist(qy)
    K = (-D2 * 0.5).exp()
    h = d - (1.0 / sigma) * ((x - qy) * py).sum() - (1.0 / sigma) * wpy * D2
    return (K * h).sum_reduction(axis=1)  # G x 1


def GaussLinKernel(sigma, d, l):
    # u and v are the feature vectors
    x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
    D2 = x.sqdist(y)
    K = (-D2 / (2.0 * sigma * sigma)).exp() * (u * v).sum()
    return (K).sum_reduction(axis=1)


def GaussKernelB(sigma, d):
    # b is px (spatial momentum)
    x, y, b = Vi(0, d) / sigma, Vj(1, d) / sigma, Vj(2, d)
    D2 = x.sqdist(y)
    K = (-D2 * 0.5).exp()
    return (K * b).sum_reduction(axis=1)


###################################################################
# Integration


def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l

    return f


#################################################################
# Hamiltonian
def Hamiltonian(K0, sigma, d, numS):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]

        # h = 0.5*Vi(px)*Vj(px).sum() - (1.0/sigma) * Vj(pw)*Vj(qw)*((Vi(qx) - Vj(qx))*Vi(px)).sum() - (1.0/(2*sigma**2)) * Vi(pw*qw)*Vj(pw*qw)*(Vi(px).sqdist(Vj(px)) - d)
        # print(h)
        """
        H0 = (px * K0(qx,qx,px)).sum()
        H1 = (px * K1(qx, qx, pw*qw)).sum()
        H2 = (K2(qx,qx,pw*qw, pw*qw)).sum()
        return 0.5 * H0 + H1 - 0.5*H2 # 0.5 * (px * K(qx, qx, px)).sum()
        """
        # return ((h*K0(qx,qx)).sum_reduction(axis=1)).sum()
        wpq = pw * qw
        print("wpq shape")
        print(wpq.detach().shape)
        k = K0(qx, qx, px, px, wpq, wpq)
        print("k is ")
        print(k.detach().cpu().numpy())
        return k.sum()

    return H


def HamiltonianSystem(K0, sigma, d, numS):
    H = Hamiltonian(K0, sigma, d, numS)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        print("Gp and Gq are ")
        print(Gp.detach().cpu().numpy())
        print(Gq.detach().cpu().numpy())
        return -Gq, Gp

    return HS


def HamiltonianSystemGrid(K0, sigma, d, numS):
    H = Hamiltonian(K0, sigma, d, numS)

    def HS(p, q, qgrid, qgridw):
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]
        gx = qgrid.view(-1, d)
        gw = qgridw.view(-1, 1)
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        print("Gp, Gq shape")
        print(Gp.detach().shape)
        print(Gq.detach().shape)
        Gg = GaussKernelU(sigma, d)(gx, qx, px, pw * qw).flatten()
        Ggw = (GaussKernelUdiv(sigma, d)(gx, qx, px, pw * qw) * gw).flatten()
        print("qgrid and qgridw shape")
        print(qgrid.detach().shape)
        print(qgridw.detach().shape)
        print("Gg and Ggw shape")
        print(Gg.detach().shape)
        print(Ggw.detach().shape)
        return -Gq, Gp, Gg, Ggw

    return HS


##################################################################
# Shooting


def Shooting(p0, q0, K0, K1, sigma, d, numS, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K0, sigma, d, numS), (p0, q0), nt)


def Flow(x0, p0, q0, K0, K1, sigma, d, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K0, sigma, d, numS)

    def FlowEq(x, p, q):
        return (K1(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K0, K1, sigma, d, numS, dataloss, gamma=1.0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K0, K1, sigma, d, numS)[-1]
        print("p,q after shooting ")
        print(p.detach().cpu().numpy())
        print(q.detach().cpu().numpy())
        return gamma * Hamiltonian(K0, sigma, d, numS)(p0, q0), dataloss(q)
        # return dataloss(q)

    return loss


def ShootingGrid(
    p0, q0, qGrid, qGridw, K0, sigma, d, numS, nt=10, Integrator=RalstonIntegrator()
):
    return Integrator(
        HamiltonianSystemGrid(K0, sigma, d, numS), (p0, q0, qGrid, qGridw), nt
    )


#################################################################
# Data Attachment Term
# K kernel for Varifold Norm (GaussLinKernel)
def lossVarifoldNorm(T, w_T, zeta_T, zeta_S, K, d, numS, beta):
    # print(w_T*zeta_T.cpu().numpy())
    cst = (K(T, T, w_T * zeta_T, w_T * zeta_T)).sum()
    print("cst is ")
    print(cst.detach().cpu().numpy())

    def loss(sS):
        # sS will be in the form of q (w_S,S)
        sSx = sS[numS:].view(-1, d)
        sSw = sS[:numS].view(-1, 1)
        print("shapes of variables")
        print(sSx.shape)
        print(sSw.shape)
        print(zeta_S.shape)
        print(zeta_T.shape)
        print(T.shape)
        print(w_T.shape)

        k1 = K(sSx, sSx, sSw * zeta_S, sSw * zeta_S)
        print("ks")
        print(k1.detach().cpu().numpy())
        k2 = K(sSx, T, sSw * zeta_S, w_T * zeta_T)
        print(k2.detach().cpu().numpy())

        return (beta / 2.0) * (cst + k1.sum() - 2 * k2.sum())

    return cst.detach().cpu().numpy(), loss


###################################################################
# Optimization


def makePQ(S, nu_S, T, nu_T):
    # initialize state vectors based on normalization
    w_S = nu_S.sum(axis=-1)[..., None].type(dtype)
    w_T = nu_T.sum(axis=-1)[..., None].type(dtype)
    zeta_S = (nu_S / w_S).type(dtype)
    zeta_T = (nu_T / w_T).type(dtype)
    numS = w_S.shape[0]
    q0 = (
        torch.cat((w_S.clone().detach().flatten(), S.clone().detach().flatten()), 0)
        .requires_grad_(True)
        .type(dtype)
    )
    print("q0 shape")
    print(q0.shape)
    p0 = (torch.zeros_like(q0)).requires_grad_(True).type(dtype)

    return w_S, w_T, zeta_S, zeta_T, q0, p0, numS


def callOptimize(
    S, nu_S, T, nu_T, sigmaRKHS, sigmaVar, d, labs, savedir, its=100, beta=0.1
):
    w_S, w_T, zeta_S, zeta_T, q0, p0, numS = makePQ(S, nu_S, T, nu_T)
    print("data types")
    print(T.dtype)
    print(w_T.dtype)
    print(zeta_T.dtype)
    cst, dataloss = lossVarifoldNorm(
        T,
        w_T,
        zeta_T,
        zeta_S,
        GaussLinKernel(sigma=sigmaVar, d=d, l=labs),
        d,
        numS,
        beta=beta,
    )
    Kg = GaussKernelHamiltonian(sigma=sigmaRKHS, d=d)
    Kv = GaussKernelB(sigma=sigmaRKHS, d=d)

    loss = LDDMMloss(Kg, Kv, sigmaRKHS, d, numS, dataloss)

    optimizer = torch.optim.LBFGS(
        [p0], max_eval=10, max_iter=10, line_search_fn="strong_wolfe"
    )
    print("performing optimization...")
    start = time.time()

    # keep track of both losses
    lossListH = []
    lossListDA = []
    relLossList = []

    def closure():
        optimizer.zero_grad()
        LH, LDA = loss(p0, q0)
        L = LH + LDA
        print("loss", L.detach().cpu().numpy())
        lossListH.append(np.copy(LH.detach().cpu().numpy()))
        lossListDA.append(np.copy(LDA.detach().cpu().numpy()))
        relLossList.append(np.copy(LDA.detach().cpu().numpy()) / cst)
        L.backward()
        return L

    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(closure)
    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    f, ax = plt.subplots()
    ax.plot(
        np.arange(len(lossListH)),
        np.asarray(lossListH),
        label="H($q_0$,$p_0$), Final = {0:.2f}".format(lossListH[-1]),
    )
    ax.plot(
        np.arange(len(lossListH)),
        np.asarray(lossListDA),
        label="Varifold Norm, Final = {0:.2f}".format(lossListDA[-1]),
    )
    ax.plot(
        np.arange(len(lossListH)),
        np.asarray(lossListDA) + np.asarray(lossListH),
        label="Total Cost, Final = {0:.2f}".format(lossListDA[-1] + lossListH[-1]),
    )
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + "Cost.png", dpi=300)

    f, ax = plt.subplots()
    ax.plot(
        np.arange(len(lossListH)),
        np.asarray(lossListDA / cst),
        label="Varifold Norm, Final = {0:.2f}".format(lossListDA[-1] / cst),
    )
    ax.set_title("Relative Loss Varifold Norm")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(savedir + "RelativeLossVarifoldNorm.png", dpi=300)

    # Print out deformed states
    # listpq = Shooting(p0, q0, Kg, Kv, sigmaRKHS,d,numS)
    coords = q0[numS:].detach().view(-1, d)
    xGrid = torch.arange(
        torch.min(coords[..., 0]) - 0.1, torch.max(coords[..., 0]) + 0.2, 0.1
    )
    yGrid = torch.arange(
        torch.min(coords[..., 1]) - 0.1, torch.max(coords[..., 1]) + 0.2, 0.1
    )
    zGrid = torch.arange(
        torch.min(coords[..., 2]) - 0.1, torch.max(coords[..., 2]) + 0.2, 0.1
    )
    XG, YG, ZG = torch.meshgrid((xGrid, yGrid, zGrid), indexing="ij")
    qGrid = torch.stack((XG.flatten(), YG.flatten(), ZG.flatten()), axis=-1).type(dtype)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG)).type(dtype)
    listpq = ShootingGrid(p0, q0, qGrid, qGridw, Kg, sigmaRKHS, d, numS)
    Dlist = []
    nu_Dlist = []
    Glist = []
    wGlist = []
    for t in range(10):
        qnp = listpq[t][1]
        D = qnp[numS:].detach().view(-1, d).cpu().numpy()
        muD = qnp.detach().cpu().numpy()
        nu_D = np.squeeze(muD[0:numS])[..., None] * zeta_S.detach().cpu().numpy()
        Dlist.append(D)
        nu_Dlist.append(nu_D)
        gt = listpq[t][2]
        G = gt.detach().view(-1, d).cpu().numpy()
        Glist.append(G)
        gw = listpq[t][3]
        W = gw.detach().cpu().numpy()
        wGlist.append(W)
        # plot p0 as arrows
    listSp0 = np.zeros((numS * 2, 3))
    polyListSp0 = np.zeros((numS, 3))
    polyListSp0[:, 0] = 2
    polyListSp0[:, 1] = np.arange(numS)  # +1
    polyListSp0[:, 2] = numS + np.arange(numS)  # + 1
    listSp0[0:numS, :] = S.detach().cpu().numpy()
    listSp0[numS:, :] = (
        p0[numS:].detach().view(-1, d).cpu().numpy() + listSp0[0:numS, :]
    )
    featsp0 = np.zeros((numS * 2, 1))
    featsp0[numS:, :] = p0[0:numS].detach().view(-1, 1).cpu().numpy()
    vtf.writeVTK(
        listSp0,
        [featsp0],
        ["p0_w"],
        savedir + "testOutput_p0.vtk",
        polyData=polyListSp0,
    )
    return Dlist, nu_Dlist, Glist, wGlist
