import torch
from pykeops.torch import Vi, Vj, LazyTensor
from xmodmap.preprocess.preprocess import rescaleData

class LossVarifoldNorm:
    """
    beta: weight for the varifold term to rescale the varifold norm to be 1.0 for each scale (see uCoeff !)
    """
    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde):

        self.sigmaVar = sigmaVar

        self.Ttilde = Ttilde
        self.w_T = w_T
        self.zeta_T = zeta_T

        self.d = Ttilde.shape[1]
        self.labs = zeta_T.shape[1]

        self.beta = [1. for _ in range(len(sigmaVar))]

        self.weight = 1. # weight coeff infront of dataloss, correspond to 1 / gamma

    @staticmethod
    def GaussLinKernelSingle(sig, d, l):
        # u and v are the feature vectors
        x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
        D2 = x.sqdist(y)
        K = (-D2 / (2.0 * sig * sig)).exp() * (u * v).sum()
        return K.sum_reduction(axis=1)

    def GaussLinKernel(self, sigma_list, beta): #sigma, d, l, beta):
        """
        \sum_sigma \beta/2 * |\mu_s - \mu_T|^2_\sigma
        """

        # u and v are the feature vectors
        x, y, u, v = Vi(0, self.d), Vj(1, self.d), Vi(2, self.labs), Vj(3, self.labs)
        D2 = x.sqdist(y)
        retVal = LazyTensor(0.)
        for sInd in range(len(self.sigmaVar)):
            sig = self.sigmaVar[sInd]
            K = (-D2 / (2.0 * sig * sig)).exp() * (u * v).sum()
            retVal += self.beta[sInd] * K

        return (retVal).sum_reduction(axis=1)

    def supportWeight(self, qx):
        # TODO: remove this to keep a vanilla implementation of varifold
        """
        no boundary estimations. Return a constant weight 1.
        """
        return torch.ones(qx.shape[0], 1)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self.update_cst_and_K()

    def normalize_across_scale(self, Stilde, weight, zeta_S, pi_STinit):
        self.set_normalization(Stilde, weight, zeta_S, pi_STinit)

    def set_normalization(self, Stilde, w_S, zeta_S, pi_STinit):
        """
        sW0 : are the weights mask for the support of the target on the source domain
        """

        # set beta to make ||mu_S - mu_T||^2 = 1
        tmp = (w_S * zeta_S) @ pi_STinit
        beta = []
        for sig in self.sigmaVar:
            Kinit = self.GaussLinKernelSingle(sig, self.d, self.labs)
            cinit = Kinit(self.Ttilde, self.Ttilde, self.w_T * self.zeta_T, self.w_T * self.zeta_T).sum()
            k1 = Kinit(Stilde, Stilde, tmp, tmp).sum()
            k2 = (- 2.0 * Kinit(Stilde, self.Ttilde, tmp, self.w_T * self.zeta_T)).sum()

            beta.append((0.6 / sig) * 2.0 / (cinit + k1 + k2))

            print("mu source norm ", k1)
            print("mu target norm ", cinit)
            print("total norm ", (cinit + k1 + k2))

        self.beta = beta

    def update_cst_and_K(self):
        self.K = self.GaussLinKernel(self.sigmaVar, self.beta)

        self.cst = self.K(self.Ttilde,
                          self.Ttilde,
                          self.w_T * self.zeta_T,
                          self.w_T * self.zeta_T).sum()

    def __call__(self, sSx, sSw, zeta_S, pi_ST):
        """
        sSw : shot Source weights
        sSx : shot Source locations
        zeta_S : Source feature distribution weights
        pi_ST : Source to Target features mapping
        """

        nu_Spi = (sSw * zeta_S) @ (pi_ST ** 2)  # Ns x L * L x F

        k1 = self.K(sSx, sSx, nu_Spi, nu_Spi)
        k2 = self.K(sSx, self.Ttilde, nu_Spi, self.w_T * self.zeta_T)

        return (1.0 / 2.0) * (self.cst + k1.sum() - 2.0 * k2.sum())


if __name__ == "__main__":
    #dataloss = LossVarifoldNorm(ldskf,dofjs,...)
    #

    n = 1000
    sigmaVar = [torch.tensor(0.1), torch.tensor(0.3)]

    Stilde = torch.randn(2 * n, 3)
    w_S = torch.randn(2 * n, 1)
    zeta_S = (torch.randn(2 * n, 2) > 0 ) + 0.

    Ttilde = torch.randn(n, 3)
    w_T = torch.randn(n, 1)
    zeta_T = (torch.randn(n, 2) > 0 ) + 0.

    pi_ST = torch.eye(2)

    loss = LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)
    loss.set_normalization(Stilde, w_S, zeta_S, pi_ST)

    print(loss(Stilde, w_S, zeta_S, pi_ST))
    print(loss.cst)