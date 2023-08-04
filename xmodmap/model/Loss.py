import torch

class Loss:
    """Vanilla LDDMM loss"""
    def __init__(self, hamiltonian, shooting, dataLoss):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss

    def __call__(self, px0, pw0, qx0, qw0, zeta_S):

        hLoss = self.hamiltonian.weight * self.hamiltonian(px0, pw0, qx0, qw0)

        _, _, qx, qw = self.shooting(px0, pw0, qx0, qw0)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx, qw, zeta_S) # TODO: implement vanilla dataloss

        return hLoss, dLoss


class CrossModalityLoss:
    """Cross-modality LDDMM loss"""

    def __init__(self, hamiltonian, shooting, dataLoss, piLoss):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss
        self.piLoss = piLoss

    def __call__(self, px0, pw0, qx0, qw0, pi_ST, zeta_S):

        hLoss = self.hamiltonian.weight * self.hamiltonian(px0, pw0, qx0, qw0)

        _, _, qx, qw = self.shooting(px0, pw0, qx0, qw0)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx, qw, zeta_S, pi_ST)

        pLoss = self.piLoss.weight * self.piLoss(qw, pi_ST)

        return hLoss, dLoss, pLoss



