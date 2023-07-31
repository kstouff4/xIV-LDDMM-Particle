import torch

from xmodmap.deformation.Shooting import Shooting
from xmodmap.deformation.Hamiltonian import Hamiltonian

class LDDMMloss:
    def __init__(self, Stilde, sigma, gamma, dataloss, piLoss, lambLoss, cA=1.0, cS=10., cT=1.0, cPi=10.0, dimEff=3, single=False):
        # TODO: avoid to pack and unpack p and q
        self.d = Stilde.shape[1]
        self.numS = Stilde.shape[0]

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

    def __call__(self, p0, q0):

        p, q = self.shoot(p0[: (self.d + 1) * self.numS], q0)[-1]

        hLoss = self.gamma * self.hamiltonian(p0[: (self.d + 1) * self.numS], q0)
        dLoss = self.dataloss(q, p0[(self.d + 1) * self.numS :])

        if self.lambLoss is not None:
            pLoss = self.gamma * self.cPi * self.piLoss(q, p0[(self.d + 1) * self.numS : -1])
            lLoss = self.gamma * self.lambLoss(p0[-1])
        else:
            pLoss = self.gamma * self.cPi * self.piLoss(q, p0[(self.d + 1) * self.numS :])
            lLoss = torch.tensor(0.)
        return hLoss, dLoss, pLoss, lLoss
