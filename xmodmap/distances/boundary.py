import torch

# boundary weight regularization
def supportRestrictionReg(eta0=torch.sqrt(torch.tensor(0.1))):
    """
    `lam` is a bandwith used to describe the transition zone between the ROI and the exterior.
    """

    def etaReg(eta):
        lamb = (eta / eta0) ** 2
        reg = lamb * torch.log(lamb) + 1.0 - lamb
        return reg

    return etaReg
