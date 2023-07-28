import torch

from xmodmap.deformation.control.nonRigid import getU, getUdiv
from xmodmap.deformation.control.affine import getATauAlpha


def hamiltonian(K0, sigma, d, numS, cA=1.0, cT=1.0, dimEff=3, single=False):
    # K0 = GaussKernelHamiltonian(x,x,px,px,w*pw,w*pw)
    def H(p, q):
        # TODO: avoid to pack and unpack p and q
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]

        wpq = pw * qw
        k = K0(qx, qx, px, px, wpq, wpq)  # k shape should be N x 1
        # px = N x 3, qx = N x 3, qw = N x 1
        A, tau, Alpha = getATauAlpha(
            px, qx, pw, qw, cA, cT, dimEff, single
        )  # getAtau( = (1.0/(2*alpha))*(px.T@(qx-qc) - (qx-qc).T@px) # should be d x d
        Anorm = (A * A).sum()
        Alphanorm = (Alpha * Alpha).sum()
        return (
            k.sum()
            + (cA / 2.0) * Anorm
            + (cT / 2.0) * (tau * tau).sum()
            + (0.5) * Alphanorm
        )

    return H


def hamiltonianSystem(K0, sigma, d, numS, cA=1.0, cT=1.0, dimEff=3, single=False):
    H = hamiltonian(K0, sigma, d, numS, cA, cT, dimEff, single)

    def HS(p, q):
        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS


def hamiltonianSystemGrid(
    K0, sigma, d, numS, uCoeff, cA=1.0, cT=1.0, dimEff=3, single=False
):
    H = hamiltonian(K0, sigma, d, numS, cA, cT, dimEff=dimEff, single=single)

    def HS(p, q, qgrid, qgridw, T=None, wT=None):
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]
        # pc = p[-d:].view(1,d)
        # qc = q[-d:].view(1,d)
        gx = qgrid.view(-1, d)
        gw = qgridw.view(-1, 1)
        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, dimEff=dimEff, single=single)
        xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))
        Gg = (
            getU(sigma, d, uCoeff)(gx, qx, px, pw * qw)
            + (gx - xc) @ A.T
            + tau
            + (gx - xc) @ Alpha
        ).flatten()
        Ggw = (
            getUdiv(sigma, d, uCoeff)(gx, qx, px, pw * qw) * gw + Alpha.sum() * gw
        ).flatten()

        if T == None:
            return -Gq, Gp, Gg, Ggw
        else:
            # TODO: check if this is actually used
            # print("including T")
            # Tx = T.view(-1, d)
            # wTw = wT.view(-1, 1)
            # Gt = (
            #     -1.0
            #     * (
            #         getU(sigma, d, uCoeff)(Tx, qx, px, pw * qw)
            #         + (Tx - xc) @ A.T
            #         + tau
            #         + (Tx - xc) @ Alpha
            #     ).flatten()
            # )
            # Gtw = (
            #     -1.0
            #     * (
            #         getUdiv(sigma, d, uCoeff)(Tx, qx, px, pw * qw) * wTw
            #         + Alpha.sum() * wTw
            #     ).flatten()
            # )
            # return -Gq, Gp, Gg, Ggw, Gt, Gtw
            # throw implementation error
            raise NotImplementedError

    return HS


def hamiltonianSystemBackwards(
    K0, sigma, d, numS, uCoeff, cA=1.0, cT=1.0, dimEff=3, single=False
):
    # initial = integration from p0 and q0 just with negative velocity field and parameters (e.g. giving same p's)
    # 06/25 = first try with just putting -px, -pw replacement only --> doesn't work (seems to give incorrect mapping)
    # 06/26 = second try with just changing hamiltonian derivatives in terms of sign (change t --> 1 - t); scale off (too big)
    # 7/01 = third try with starting with -p1, but keeping same integration scheme in terms of relation with hamiltonians
    H = hamiltonian(K0, sigma, d, numS, cA, cT, dimEff=dimEff, single=single)

    def HS(p, q, T, wT):
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]
        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, dimEff=dimEff, single=single)
        xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))
        Tx = T.view(-1, d)
        wTw = wT.view(-1, 1)
        Gt = (
            getU(sigma, d, uCoeff)(Tx, qx, px, pw * qw)
            + (Tx - xc) @ A.T
            + tau
            + (Tx - xc) @ Alpha
        ).flatten()
        Gtw = (
            getUdiv(sigma, d, uCoeff)(Tx, qx, px, pw * qw) * wTw + Alpha.sum() * wTw
        ).flatten()
        return -Gq, Gp, Gt, Gtw

    return HS
