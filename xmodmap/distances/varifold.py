import torch
from pykeops.torch import Vi, Vj, LazyTensor

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

        # supportWeight is a function that takes as input cordiantes (ie points of Stilde) and lamb0 (a bandwith
        # of some tanh sigmoid function) and returns the support weight
        #self.supportWeights = self.defineSupport()


        self.beta = [1. for _ in range(len(sigmaVar))]


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
        self.K = self.GaussLinKernel(self.sigmaVar, self._beta)

        self.cst = self.K(self.Ttilde,
                          self.Ttilde,
                          self.w_T * self.zeta_T,
                          self.w_T * self.zeta_T).sum()


    def eval(self, sSx, sSw, zeta_S, pi_ST):
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

    def __call__(self, sSx, sSw, zeta_S, pi_ST):
        return self.eval(sSx, sSw, zeta_S, pi_ST)



class LossVarifoldNormBoundary(LossVarifoldNorm):
    """
    lamb : mask for the support of the varifold
    """
    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde, lamb0):
        """
        This function defined the support of ROI as a zone of  the ambient space. It is actually a neighborhood of the
         *target* (defined by a sigmoid function, tanh). Coefficient alpha is though as 'transparency coefficient'.

        It involves the estimation of a normal vectors and points assuming parallel planes of data (coronal sections)

        Args:
            `self.Ttilde`: target location *scaled* in a unit box (N_target x 3)

        Return:
            `alphaSupportWeight
        """
        super().__init__(sigmaVar, w_T, zeta_T, Ttilde)

    def definedSlicedSupport(self, eps=1e-3):
        zMin = torch.min(self.Ttilde[:, -1])
        zMax = torch.max(self.Ttilde[:, -1])

        print("zMin: ", zMin)
        print("zMax: ", zMax)

        if zMin == zMax:
            zMin = torch.min(self.Ttilde[:, -2])
            zMax = torch.max(self.Ttilde[:, -2])
            print("new Zmin: ", zMin)
            print("new Zmax: ", zMax)
            sh = 0
            co = 1
            while sh < 2:
                lineZMin = self.Ttilde[
                    torch.squeeze(
                        self.Ttilde[:, -2] < (zMin + torch.tensor(co * eps))
                    ),
                    ...,
                ]
                lineZMax = self.Ttilde[
                    torch.squeeze(
                        self.Ttilde[:, -2] > (zMax - torch.tensor(co * eps))
                    ),
                    ...,
                ]
                sh = min(lineZMin.shape[0], lineZMax.shape[0])
                co += 1

            print("lineZMin: ", lineZMin)
            print("lineZMax: ", lineZMax)

            tCenter = torch.mean(self.Ttilde, axis=0)

            a0s = lineZMin[torch.randperm(lineZMin.shape[0])[0:2], ...]
            a1s = lineZMax[torch.randperm(lineZMax.shape[0])[0:2], ...]

            print("a0s: ", a0s)
            print("a1s: ", a1s)

            a0 = torch.mean(a0s, axis=0)
            a1 = torch.mean(a1s, axis=0)

            print("a0: ", a0)
            print("a1: ", a1)

            n0 = torch.tensor(
                [-(a0s[1, 1] - a0s[0, 1]), (a0s[1, 0] - a0s[0, 0]), self.Ttilde[0, -1]]
            )
            n1 = torch.tensor(
                [-(a1s[1, 1] - a1s[0, 1]), (a1s[1, 0] - a1s[0, 0]), self.Ttilde[0, -1]]
            )
            if torch.dot(tCenter - a0, n0) < 0:
                n0 = -1.0 * n0

            if torch.dot(tCenter - a1, n1) < 0:
                n1 = -1.0 * n1

        else:
            sh = 0
            co = 1
            while sh < 3:
                sliceZMin = self.Ttilde[
                    torch.squeeze(
                        self.Ttilde[:, -1] < (zMin + torch.tensor(co * eps))
                    ),
                    ...,
                ]
                sliceZMax = self.Ttilde[
                    torch.squeeze(
                        self.Ttilde[:, -1] > (zMax - torch.tensor(co * eps))
                    ),
                    ...,
                ]
                sh = min(sliceZMin.shape[0], sliceZMax.shape[0])
                co += 1

            print("sliceZMin: ", sliceZMin)
            print("sliceZMax: ", sliceZMax)

            tCenter = torch.mean(self.Ttilde, axis=0)

            # pick 3 points on each approximate slice and take center and normal vector
            a0s = sliceZMin[torch.randperm(sliceZMin.shape[0])[0:3], ...]
            a1s = sliceZMax[torch.randperm(sliceZMax.shape[0])[0:3], ...]

            print("a0s: ", a0s)
            print("a1s: ", a1s)

            a0 = torch.mean(a0s, axis=0)
            a1 = torch.mean(a1s, axis=0)

            n0 = torch.cross(a0s[1, ...] - a0s[0, ...], a0s[2, ...] - a0s[0, ...])
            if torch.dot(tCenter - a0, n0) < 0:
                n0 = -1.0 * n0

            n1 = torch.cross(a1s[1, ...] - a1s[0, ...], a1s[2, ...] - a1s[0, ...])
            if torch.dot(tCenter - a1, n1) < 0:
                n1 = -1.0 * n1

        # normalize vectors
        n0 = n0 / torch.sqrt(torch.sum(n0 ** 2))
        n1 = n1 / torch.sqrt(torch.sum(n1 ** 2))

        # ensure dot product with barycenter vector to point is positive, otherwise flip sign of normal
        print("n0: ", n0)
        print("n1: ", n1)

        return n0, n1, a0, a1

    def supportWeight(self, qx, lamb):
        n0, n1, a0, a1 = self.definedSlicedSupport()

        return (0.5 * torch.tanh(torch.sum((qx - a0) * n0, axis=-1) / lamb)
                + 0.5 * torch.tanh(torch.sum((qx - a1) * n1, axis=-1) / lamb))[..., None]


    def normalize_across_scale(self, Stilde, weight, zeta_S, pi_STinit, lamb0):
        self.set_normalization(Stilde, weight * self.supportWeight(Stilde, lamb0), zeta_S, pi_STinit)

    def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb):
        return self.eval(sSx, sSw * self.supportWeight(sSx, lamb ** 2), zeta_S, pi_ST)


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