import time
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


from xmodmap.model.Model import Model

class SingleModality(Model):
    """Loss for single-modality LDDMM"""

    def __init__(self, hamiltonian, shooting, dataLoss):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss

    def get_params(self):
        params_dict = {
            "hamiltonian": self.hamiltonian.get_params(),
            "dataLoss": self.dataLoss.get_params(),
        }
        return params_dict

    def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

        _, _, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, torch.eye(zeta_S.shape[-1]))


        print(f"H loss: {hLoss.detach().cpu().numpy()}; "
              f"Var loss: {dLoss.detach().cpu().numpy()}; "
              )

        return hLoss, dLoss


    def check_resume(self, checkpoint):
        assert self.dataLoss.get_params() == checkpoint["dataLoss"]
        assert self.hamiltonian.get_params() == checkpoint["hamiltonian"]
    
    def print_log(self, logScale=False):
        # split logs
        logH = []
        logD = []

        for i in range(0,len(self.log)):
            logH.append(self.log[i][0].numpy())
            logD.append(self.log[i][1].numpy())
        
        if not logScale:
            f,ax = plt.subplots()
            ax.plot(logH, label=f"Hamiltonian Loss, Final = {logH[-1]}")
            ax.plot(logD, label=f"Data Loss, Final = {logD[-1]}")
            ax.legend()
        else:
            f,ax = plt.subplots()
            ax.plot(np.log(np.asarray(logH)+1.0), label=f"Hamiltonian Loss, Final = {logH[-1]}")
            ax.plot(np.log(np.asarray(logD)+1.0), label=f"Data Loss, Final = {logD[-1]}")
            ax.legend()
            
        
        return f



