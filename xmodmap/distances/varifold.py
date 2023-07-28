import torch
from pykeops.torch import Vi, Vj

class LossVarifoldNorm:
    def __init__(self, beta, sigmaVar, d, labs, w_S, w_T, zeta_S, zeta_T, pi_STinit, Stilde, Ttilde, lamb0):
        self.Ttilde = Ttilde
        self.w_T = w_T
        self.zeta_T = zeta_T
        self.zeta_S = zeta_S
        self.d = d
        self.labs = labs
        self.numS = Stilde.shape[0]
        self.sigmaVar = sigmaVar

        if lamb0 < 0:
            self.supportWeight = None
            sW0 = torch.ones((Stilde.shape[0], 1))
        else:
            self.supportWeight = self.defineSupport()
            sW0 = self.supportWeight(Stilde, lamb0)

        if beta is None:
            # set beta to make ||mu_S - mu_T||^2 = 1
            if len(sigmaVar) == 1:
                Kinit = self.GaussLinKernelSingle(sig=sigmaVar[0], d=d, l=labs)
                cinit = Kinit(Ttilde, Ttilde, w_T * zeta_T, w_T * zeta_T).sum()
                k1 = Kinit(
                    Stilde,
                    Stilde,
                    (sW0 * w_S * zeta_S) @ pi_STinit,
                    (sW0 * w_S * zeta_S) @ pi_STinit,
                )
                k2 = Kinit(Stilde, Ttilde, (sW0 * w_S * zeta_S) @ pi_STinit, w_T * zeta_T)
                beta = 2.0 / (cinit + k1.sum() - 2.0 * k2.sum())
                print("beta is ", beta)
                beta = [
                    (0.6 / sigmaVar[0])
                    * torch.clone(2.0 / (cinit + k1.sum() - 2.0 * k2.sum()))
                ]

            # print out indiviual costs
            else:
                print("different varifold norm at beginning")
                beta = []
                for sig in sigmaVar:
                    print("sig is ", sig)
                    Kinit = self.GaussLinKernelSingle(sig=sig, d=d, l=labs)
                    cinit = Kinit(Ttilde, Ttilde, w_T * zeta_T, w_T * zeta_T).sum()
                    k1 = Kinit(
                        Stilde,
                        Stilde,
                        (sW0 * w_S * zeta_S) @ pi_STinit,
                        (sW0 * w_S * zeta_S) @ pi_STinit,
                    ).sum()
                    k2 = (
                            -2.0
                            * Kinit(
                        Stilde, Ttilde, (sW0 * w_S * zeta_S) @ pi_STinit, w_T * zeta_T
                    ).sum()
                    )
                    beta.append(
                        (0.6 / sig) * torch.clone(2.0 / (cinit + k1 + k2))
                    )
                    print("mu source norm ", k1)
                    print("mu target norm ", cinit)
                    print("total norm ", (cinit + k1 + k2))

        self.beta = beta

        self.K = self.GaussLinKernel()

        self.cst = self.K(self.Ttilde,
                          self.Ttilde,
                          self.w_T * self.zeta_T,
                          self.w_T * self.zeta_T).sum()
    @staticmethod
    def GaussLinKernelSingle(sig, d, l):
        # u and v are the feature vectors
        x, y, u, v = Vi(0, d), Vj(1, d), Vi(2, l), Vj(3, l)
        D2 = x.sqdist(y)
        K = (-D2 / (2.0 * sig * sig)).exp() * (u * v).sum()
        return K.sum_reduction(axis=1)

    def GaussLinKernel(self): #sigma, d, l, beta):
        """
        \sum_sigma \beta/2 * |\mu_s - \mu_T|^2_\sigma
        """

        # u and v are the feature vectors
        x, y, u, v = Vi(0, self.d), Vj(1, self.d), Vi(2, self.labs), Vj(3, self.labs)
        D2 = x.sqdist(y)
        for sInd in range(len(self.sigmaVar)):
            sig = self.sigmaVar[sInd]
            K = (-D2 / (2.0 * sig * sig)).exp() * (u * v).sum()
            if sInd == 0:
                retVal = self.beta[sInd] * K
            else:
                retVal += self.beta[sInd] * K
        return (retVal).sum_reduction(axis=1)

    def defineSupport(self, eps=0.001):
        """
        This function defined the support of ROI as a zone of  the ambient space. It is actually a neighborhood of the
         *target* (defined by a sigmoid function, tanh). Coefficient alpha is though as 'transparency coefficient'.

        It involves the estimation of a normal vectors and points assuming parallel planes of data (coronal sections)

        Args:
            `self.Ttilde`: target location *scaled* in a unit box (N_target x 3)

        Return:
            `alphaSupportWeight
        """
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
        n0 = n0 / torch.sqrt(torch.sum(n0**2))
        n1 = n1 / torch.sqrt(torch.sum(n1**2))

        # ensure dot product with barycenter vector to point is positive, otherwise flip sign of normal
        print("n0: ", n0)
        print("n1: ", n1)

        def alphaSupportWeight(qx, lamb):
            return (
                0.5 * torch.tanh(torch.sum((qx - a0) * n0, axis=-1) / lamb)
                + 0.5 * torch.tanh(torch.sum((qx - a1) * n1, axis=-1) / lamb)
            )[..., None]

        return alphaSupportWeight

    def __call__(self, sS, pi_est):
        # sS will be in the form of q (w_S,S,x_c)
        sSx = sS[self.numS:].view(-1, self.d)
        sSw = sS[:self.numS].view(-1, 1)
        if self.supportWeight is not None:
            wSupport = self.supportWeight(sSx, pi_est[-1] ** 2)
            sSw = wSupport * sSw
            pi_ST = (pi_est[:-1].view(self.zeta_S.shape[-1], self.zeta_T.shape[-1])) ** 2
        else:
            pi_ST = (pi_est.view(self.zeta_S.shape[-1], self.zeta_T.shape[-1])) ** 2
        nu_Spi = (sSw * self.zeta_S) @ pi_ST  # Ns x L * L x F

        k1 = self.K(sSx, sSx, nu_Spi, nu_Spi)
        k2 = self.K(sSx, self.Ttilde, nu_Spi, self.w_T * self.zeta_T)

        return (1.0 / 2.0) * (self.cst + k1.sum() - 2.0 * k2.sum())
