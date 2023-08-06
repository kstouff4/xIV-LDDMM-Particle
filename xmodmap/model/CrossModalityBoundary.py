import time

import torch
class CrossModalityBoundary:
    """Boundary loss for cross-modality LDDMM"""

    def __init__(self, hamiltonian, shooting, dataLoss, piLoss, lambLoss, variables, variables_to_optimize, pwTilde=None, pxTilde=None, Csqpi=None, Csqlamb=None):
        self.hamiltonian = hamiltonian
        self.shooting = shooting
        self.dataLoss = dataLoss
        self.piLoss = piLoss
        self.lambLoss = lambLoss

        # store the ptilde variable, TODO: move this as a preconditionner
        self.pwTilde = pwTilde
        self.pxTilde = pxTilde
        self.Csqpi = Csqpi
        self.Csqlamb = Csqlamb

        self.variables = variables
        self.variables_to_optimize = [self.variables[i] for i in variables_to_optimize]

        self.lossListH = []
        self.lossListDA = []
        self.lossListPI = []
        self.lossListL = []


    def loss(self, px0, pw0, qx0, qw0, pi_ST, zeta_S, lamb):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px0, pw0, qx0, qw0)

        _, _, qx, qw = self.shooting(px0, pw0, qx0, qw0)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx, qw, zeta_S, pi_ST, lamb)

        pLoss = self.piLoss.weight * self.piLoss(qw, pi_ST)

        lLoss = self.lambLoss.weight * self.lambLoss(lamb)

        return hLoss, dLoss, pLoss, lLoss

    def closure(self):
        self.optimizer.zero_grad()
        hLoss, dLoss, pLoss, lLoss = self.loss(self.variables["px"] * self.pxTilde,
                                               self.variables["pw"] * self.pwTilde,
                                               self.variables["qx"],
                                               self.variables["qw"],
                                               self.variables["pi_ST"] * self.Csqpi,
                                               self.variables["zeta_S"],
                                               self.variables["lamb"] * self.Csqlamb)
        loss = hLoss + dLoss + pLoss + lLoss

        # move the value to cpu() to be matplotlib compatible
        self.lossListH.append(hLoss.detach().clone().cpu())
        self.lossListDA.append(dLoss.detach().clone().cpu())
        self.lossListPI.append(pLoss.detach().clone().cpu())
        self.lossListL.append(lLoss.detach().clone().cpu())

        loss.backward()
        return loss

    def optimize(self,max_eval=15, max_iter=100,
            line_search_fn="strong_wolfe",
            history_size=100,
            tolerance_grad=1e-8,
            tolerance_change=1e-10):

        self.optimizer = torch.optim.LBFGS(
            self.variables_to_optimize,
            max_eval=max_eval,
            max_iter=max_iter,
            line_search_fn=line_search_fn,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change
        )

        print("performing optimization...")
        start = time.time()

        for i in range(max_iter):
            print("it ", i, ": ", end="")
            self.optimizer.step(self.closure)

            # print current losses values
            print("Current Losses", flush=True)
            print("H loss: ", self.lossListH[-1])
            print("Var loss: ", self.lossListDA[-1])
            print("Pi lossL ", self.lossListPI[-1])
            print("Lambda loss ", self.lossListL[-1])

            # check for NaN
            osd = self.optimizer.state_dict()

            assert not torch.isnan(self.lossListH[-1]), "H loss is NaN" + str(osd)
            assert not torch.isnan(self.lossListDA[-1]), "Var loss is NaN" + str(osd)
            assert not torch.isnan(self.lossListPI[-1]), "Pi loss is NaN" + str(osd)
            assert not torch.isnan(self.lossListL[-1]), "Lambda loss is NaN" + str(osd)

        print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")






