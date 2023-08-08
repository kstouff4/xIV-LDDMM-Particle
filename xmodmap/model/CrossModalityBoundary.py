import logging.handlers
import os
import time
import torch

from xmodmap.optimizer.config import lbfgsConfigDefaults

class CrossModalityBoundary:
    """Boundary loss for cross-modality LDDMM"""

    variables = None
    variables_to_optimize = None
    precond = lambda *x: x  # identity function
    optimizer = None
    log = []
    current_step = 0
    _savedir = None

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

    def init(self, variables, variables_to_optimize, precond=None, savedir=os.path.join(os.getcwd(), "output")):
        # a dict containing all the variables of the model
        self.variables = variables

        # a list of the variables to optimize
        self.variables_to_optimize = {key: self.variables[key] for i, key in enumerate(variables_to_optimize)}

        # no precond by default, see self.set_precond()
        if precond is not None:
            self.set_precond(precond)

        # init the optimizer
        self.set_optimizer()

        # init the logger
        self.savedir = savedir

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        self._savedir = savedir

    def set_optimizer(self, state=None):
        self.optimizer = torch.optim.LBFGS(self.variables_to_optimize.values(), **lbfgsConfigDefaults)
        if state is not None:
            self.optimizer.load_state_dict(state)
    def set_precond(self, weights=None):
        # check if the keys of kwargs are in self.variables
        assert weights.keys() <= self.variables.keys()

        self.precondWeights = weights
        # init a dict with value 1
        precond = {key: 1.0 for key in self.variables}
        # update the dict with the kwargs
        precond.update(weights)

        # create a function that returns the precond for each variable
        self.precond = lambda x: {key: x[key] * precond[key] for key in self.variables}

    def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None, lamb=None):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

        _, _, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]
        dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST, lamb)

        pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)

        lLoss = self.lambLoss.weight * self.lambLoss(lamb)

        return hLoss, dLoss, pLoss, lLoss

    def closure(self):
        self.optimizer.zero_grad()
        losses = self.loss(**self.precond(self.variables))

        loss = sum(losses)
        self.log.append([l.detach().cpu() for l in losses])

        print(f"H loss: {self.log[-1][0].numpy()}; "
              f"Var loss: {self.log[-1][1].numpy()}; "
              f"Pi loss: {self.log[-1][2].numpy()}; "
              f"Lambda loss: {self.log[-1][3].numpy()}")

        loss.backward()
        return loss

    def optimize(self, steps=100):

        self.steps = steps

        print("performing optimization...")
        start = time.time()

        for i in range(self.current_step, self.current_step + self.steps):
            print("it ", i, ": ", end="")
            self.optimizer.step(self.closure)

            if self.stoppingCondition():
                break

            self.current_step += 1
            self.saveState()


        print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")


    def stoppingCondition(self):
        # check for NaN
        if any(torch.isnan(l) for l in self.log[-1]):
            print(f"NaN encountered at iteration {self.current_step}.")

        # check for convergence
        if (self.current_step > 0) and torch.allclose(torch.tensor(self.log[-1]),
                                                      torch.tensor(self.log[-2]),
                                                      atol=1e-6, rtol=1e-5):
            print(f"Local minimum reached at iteration {self.current_step}.")

        if self.current_step == self.steps:
            print(f"Maximum number {self.steps} of iterations reached.")

        return 0

    def saveState(self):
        """
        osd = state of optimizer
        its = total iterations
        i = current iteration
        xopt = current optimization variable (p0*pTilde)
        """

        checkpoint = {
            "variables_to_optimize": self.variables_to_optimize,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "precondWeights": self.precondWeights,
            "steps": self.steps,
            "current_step": self.current_step,
            "log": self.log,
            "savedir": self.savedir,
        }
        checkpoint.update(self.get_params())

        filename = os.path.join(self.savedir, f"checkpoint.pt")
        torch.save(checkpoint, filename)

    def resume(self, variables, filename):

        print(f"Resuming optimization from {filename}. Loading... ", end="")

        checkpoint = torch.load(filename)

        self.variables = variables
        self.variables.update(checkpoint["variables_to_optimize"])
        self.variables_to_optimize = checkpoint["variables_to_optimize"]

        self.set_optimizer(state=checkpoint["optimizer_state_dict"])
        self.set_precond(weights=checkpoint["precondWeights"])

        self.steps = checkpoint["steps"]
        self.current_step = checkpoint["current_step"]
        self.log = checkpoint["log"]
        self.savedir = checkpoint["savedir"]

        assert self.dataLoss.get_params() == checkpoint["dataLoss"]
        assert self.hamiltonian.get_params() == checkpoint["hamiltonian"]
        assert self.piLoss.get_params() == checkpoint["piLoss"]
        assert self.lambLoss.get_params() == checkpoint["lambLoss"]

        print("done.")