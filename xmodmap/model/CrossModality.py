import time
import torch


from xmodmap.model.Model import Model

class CrossModality(Model):
    """Loss for cross-modality LDDMM"""

    def __init__(self, hamiltonian, shooting, dataLoss, piLoss):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss
        self.piLoss = piLoss
        #setattr(self, f"{key}Precond", value)

    def get_params(self):
        params_dict = {
            "hamiltonian": self.hamiltonian.get_params(),
            "dataLoss": self.dataLoss.get_params(),
            "piLoss": self.piLoss.get_params(),
        }
        return params_dict

    def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

        _, _, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST)

        pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)


        print(f"H loss: {hLoss.detach().cpu().numpy()}; "
              f"Var loss: {dLoss.detach().cpu().numpy()}; "
              f"Pi loss: {pLoss.detach().cpu().numpy()}; "
              )

        return hLoss, dLoss, pLoss


    def check_resume(self, checkpoint):
        assert self.dataLoss.get_params() == checkpoint["dataLoss"]
        assert self.hamiltonian.get_params() == checkpoint["hamiltonian"]
        assert self.piLoss.get_params() == checkpoint["piLoss"]

