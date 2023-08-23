import time
import torch
from matplotlib import pyplot as plt


from xmodmap.model.Model import Model

class CrossModalityBoundary(Model):
    """Boundary loss for cross-modality LDDMM"""

    def __init__(self, hamiltonian, shooting, dataLoss, piLoss, lambLoss):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss
        self.piLoss = piLoss
        self.lambLoss = lambLoss
        #setattr(self, f"{key}Precond", value)

    def get_params(self):
        params_dict = {
            "hamiltonian": self.hamiltonian.get_params(),
            "dataLoss": self.dataLoss.get_params(),
            "piLoss": self.piLoss.get_params(),
            "lambLoss": self.lambLoss.get_params(),
        }
        return params_dict

    def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None, lamb=None):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

        _, _, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST, lamb)

        pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)

        lLoss = self.lambLoss.weight * self.lambLoss(lamb)

        print(f"H loss: {hLoss.detach().cpu().numpy()}; "
              f"Var loss: {dLoss.detach().cpu().numpy()}; "
              f"Pi loss: {pLoss.detach().cpu().numpy()}; "
              f"Lambda loss: {lLoss.detach().cpu().numpy()}")

        return hLoss, dLoss, pLoss, lLoss


    def check_resume(self, checkpoint):
        assert self.dataLoss.get_params() == checkpoint["dataLoss"]
        assert self.hamiltonian.get_params() == checkpoint["hamiltonian"]
        assert self.piLoss.get_params() == checkpoint["piLoss"]
        assert self.lambLoss.get_params() == checkpoint["lambLoss"]
    
    def print_log(self, logScale=False):
        # split logs
        logH = []
        logD = []
        logP = []
        logL = []

        for i in range(0,len(self.log)):
            logH.append(self.log[i][0].numpy())
            logD.append(self.log[i][1].numpy())
            logP.append(self.log[i][2].numpy())
            logL.append(self.log[i][3].numpy())
        
        if not logScale:
            f,ax = plt.subplots()
            ax.plot(logH, label=f"Hamiltonian Loss, Final = {logH[-1]}")
            ax.plot(logD, label=f"Data Loss, Final = {logD[-1]}")
            ax.plot(logP, label=f"Pi Loss, Final = {logP[-1]}")
            ax.plot(logL, label=f"Lambda Loss, Final = {logL[-1]}")
            ax.legend()
        else:
            f,ax = plt.subplots()
            ax.plot(np.log(np.asarray(logH)), label=f"Hamiltonian Loss, Final = {logH[-1]}")
            ax.plot(np.log(np.asarray(logD)), label=f"Data Loss, Final = {logD[-1]}")
            ax.plot(np.log(np.asarray(logP)), label=f"Pi Loss, Final = {logP[-1]}")
            ax.plot(np.log(np.asarray(logL)), label=f"Lambda Loss, Final = {logL[-1]}")
            ax.legend()
            
        
        return f

