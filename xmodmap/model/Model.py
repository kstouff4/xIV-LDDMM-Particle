import os
import time
import torch

from xmodmap.optimizer.config import lbfgsConfigDefaults


class Model:
    variables = None
    variables_to_optimize = None
    precond = lambda *x: x  # identity function
    optimizer = None
    log = []
    current_step = 0
    _savedir = None

    def __init__(self, *args):
        pass

    def get_params(self):
        pass

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

    def closure(self):
        self.optimizer.zero_grad()
        losses = self.loss(**self.precond(self.variables))

        loss = sum(losses)
        self.log.append([l.detach().cpu() for l in losses])

        loss.backward()
        return loss

    def optimize(self, steps=100):

        self.steps = steps

        print("performing optimization...")
        start = time.time()

        for i in range(self.current_step, self.current_step + self.steps):
            print("it ", i, ": ", end="")
            self.optimizer.step(self.closure)

            self.saveState()

            if self.stoppingCondition():
                break

            self.current_step += 1

        print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    def stoppingCondition(self):
        # check for NaN
        if any(torch.isnan(l) for l in self.log[-1]):
            print(f"NaN encountered at iteration {self.current_step}.")
            return 1

        # check for convergence
        if (self.current_step > 0) and torch.allclose(torch.tensor(self.log[-1]),
                                                      torch.tensor(self.log[-2]),
                                                      atol=1e-6, rtol=1e-5):
            print(f"Local minimum reached at iteration {self.current_step}.")
            return 1

        if self.current_step == self.steps:
            print(f"Maximum number {self.steps} of iterations reached.")
            return 1

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

        self.check_resume(checkpoint)

        print("done.")


    def check_resume(self, checkpoint):
        pass
