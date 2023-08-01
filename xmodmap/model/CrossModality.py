import torch

from xmodmap.deformation.Shooting import Shooting
from xmodmap.deformation.Hamiltonian import Hamiltonian

class LDDMMloss:
    def __init__(self, Stilde, sigma, gamma, dataloss, piLoss, lambLoss, cA=1.0, cS=10., cT=1.0, cPi=10.0, dimEff=3, single=False):

        self.Stilde = Stilde
        self.sigma = sigma
        self.gamma = gamma
        self.dataloss = dataloss
        self.piLoss = piLoss
        self.lambLoss = lambLoss
        self.cA = cA
        self.cS = cS
        self.cT = cT
        self.cPi = cPi
        self.dimEff = dimEff
        self.single = single

        self.hamiltonian = Hamiltonian(sigma,
                                       Stilde,
                                       cA=cA,
                                       cS=cS,
                                       cT=cT,
                                       dimEff=dimEff,
                                       single=single)

        self.shoot = Shooting(sigma,
                              Stilde,
                              cA=cA,
                              cS=cS,
                              cT=cT,
                              dimEff=dimEff,
                              single=single)

    def __call__(self, px0, pw0, qx0, qw0, pi_ST, zeta_S):

        px, pw, qx, qw = self.shoot(px0, pw0, qx0, qw0)[-1]

        hLoss = self.gamma * self.hamiltonian(px0, pw0, qx0, qw0)
        dLoss = self.dataloss(qx, qw, zeta_S, pi_ST)

        if self.lambLoss is not None:
            pLoss = self.gamma * self.cPi * self.piLoss(qw, pi_ST)
            lLoss = self.gamma * self.lambLoss(p0[-1])
        else:
            pLoss = self.gamma * self.cPi * self.piLoss(qw, pi_ST)
            lLoss = torch.tensor(0.)
        return hLoss, dLoss, pLoss, lLoss
