import torch

from pykeops.torch import Vi, Vj

from xmodmap.deformation.control.nonRigid import getU, getUdiv
from xmodmap.deformation.control.affine import getATauAlpha
import xmodmap.deformation.Ucoeff as uc


class Hamiltonian:
    def __init__(self, sigma, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=3, single=False):
        # TODO: avoid to pack and unpack p and q
        self.d = Stilde.shape[1]
        self.numS = Stilde.shape[0]

        self.sigma = sigma
        # coefficients for the Hamiltonian that weight the different kind of control
        self.cA = cA  # rotation A
        self.cS = cS  # non rigid U
        self.cT = cT  # translation tau
        self.dimEff = dimEff
        self.single = single

        # Compute constants to weigh each kernel norm in RKHS by
        self.uCoeff = uc.Ucoeff(self.sigma, Stilde).uCoeff
        self.K0 = self.GaussKernelHamiltonian(self.sigma, self.d, self.uCoeff)

        # coefficient use in the final loss
        self.weight = 1.0 # == gamma

    def GaussKernelHamiltonian(self, sigma, d, uCoeff):
        qxO = Vi(0, d)
        qyO = Vj(1, d)
        px = Vi(2, d)
        py = Vj(3, d)
        wpxO = Vi(4, 1)
        wpyO = Vj(5, 1)

        for sInd in range(len(sigma)):
            sig = sigma[sInd]
            qx, qy, wpx, wpy = qxO / sig, qyO / sig, wpxO / sig, wpyO / sig
            D2 = qx.sqdist(qy)
            K = (-D2 * 0.5).exp()
            h = (
                    0.5 * (px * py).sum()
                    + wpy * ((qx - qy) * px).sum()
                    - (0.5) * wpx * wpy * (D2 - d)
            )  # 1/2 factor included here
            if sInd == 0:
                retVal = (1.0 / uCoeff[sInd]) * K * h  # normalize by N_sigma/(N sigma**2)
            else:
                retVal += (1.0 / uCoeff[sInd]) * K * h

        return retVal.sum_reduction(
            axis=1
        )  # (K*h).sum_reduction(axis=1) #,  h2, h3.sum_reduction(axis=1)

    def H(self, px, pw, qx, qw):
        wpq = pw * qw
        k = self.K0(qx, qx, px, px, wpq, wpq)  # k shape should be N x 1
        # px = N x 3, qx = N x 3, qw = N x 1
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, cA=self.cA, cT=self.cT, dimEff=self.dimEff, single=self.single)
        Anorm = (A * A).sum()
        Alphanorm = (Alpha * Alpha).sum()
        return (
            self.cS * k.sum()
            + (self.cA / 2.0) * Anorm
            + (self.cT / 2.0) * (tau * tau).sum()
            + (0.5) * Alphanorm
        )

    def __call__(self, px, pw, qx, qw):
        return self.H(px, pw, qx, qw)


class HamiltonianSystem(Hamiltonian):
    def __init__(self, sigma, Stilde, cA=1.0, cS=10.0, cT=1.0, dimEff=3, single=False):
        super().__init__(sigma, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)

    def __call__(self, px, pw, qx, qw):
        Gpx, Gpw, Gqx, Gqw = torch.autograd.grad(self.H(px, pw, qx, qw), (px, pw, qx, qw), create_graph=True)
        return -Gqx, -Gqw, Gpx, Gpw


class HamiltonianSystemGrid(Hamiltonian):
    def __init__(self, sigma, Stilde, cA=1.0, cS=10.0, cT=1.0, dimEff=3, single=False):
        super().__init__(sigma, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)

    def __call__(self, px, pw, qx, qw, qgrid, qgridw):
        gx = qgrid.view(-1, self.d)
        gw = qgridw.view(-1, 1)

        Gpx, Gpw, Gqx, Gqw = torch.autograd.grad(self.H(px, pw, qx, qw), (px, pw, qx, qw), create_graph=True)
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, dimEff=self.dimEff, single=self.single)
        xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))
        Gg = (
            getU(self.sigma, self.d, self.uCoeff)(gx, qx, px, pw * qw)
            + (gx - xc) @ A.T
            + tau
            + (gx - xc) @ Alpha
        )
        Ggw = (
            getUdiv(self.sigma, self.d, self.uCoeff)(gx, qx, px, pw * qw) * gw
            + Alpha.sum() * gw
        )

        return -Gqx, -Gqw, Gpx, Gpw, Gg, Ggw


class HamiltonianSystemBackwards(Hamiltonian):
    """
    initial = integration from p0 and q0 just with negative velocity field and parameters (e.g. giving same p's)
    06/25 = first try with just putting -px, -pw replacement only --> doesn't work (seems to give incorrect mapping)
    06/26 = second try with just changing hamiltonian derivatives in terms of sign (change t --> 1 - t); scale off (too big)
    7/01 = third try with starting with -p1, but keeping same integration scheme in terms of relation with hamiltonians
    """

    def __init__(self, sigma, Stilde, cA=1.0, cS=10.0, cT=1.0, dimEff=3, single=False):
        super().__init__(sigma, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)

    def __call__(self, px, pw, qx, qw, T, wT):

        Gpx, Gpw, Gqx, Gqw = torch.autograd.grad(self.H(px, pw, qx, qw), (px, pw, qx, qw), create_graph=True)
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, dimEff=self.dimEff, single=self.single)
        xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))
        Tx = T.view(-1, self.d)
        wTw = wT.view(-1, 1)
        Gt = (
            getU(self.sigma, self.d, self.uCoeff)(Tx, qx, px, pw * qw)
            + (Tx - xc) @ A.T
            + tau
            + (Tx - xc) @ Alpha
        )
        Gtw = (
            getUdiv(self.sigma, self.d, self.uCoeff)(Tx, qx, px, pw * qw) * wTw + Alpha.sum() * wTw
        )
        return -Gqx, -Gqw, Gpx, Gpw, Gt, Gtw
