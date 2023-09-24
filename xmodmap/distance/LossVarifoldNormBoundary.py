import torch

from xmodmap.distance.LossVarifoldNorm import LossVarifoldNorm


class LossVarifoldNormBoundary(LossVarifoldNorm):
    """
    lamb : mask for the support of the varifold
    """
    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde):
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
        self.n0, self.n1, self.a0, self.a1 = self.definedSlicedSupport()

    def definedSlicedSupport(self, eps=1e-3):
        if self.Ttilde.shape[-1] < 3:
            return torch.zeros((1,2)), torch.zeros((1,2)), torch.zeros((1,2)), torch.zeros((1,2))
        
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

        return (0.5 * torch.tanh(torch.sum((qx - self.a0) * self.n0, axis=-1) / lamb)
                + 0.5 * torch.tanh(torch.sum((qx - self.a1) * self.n1, axis=-1) / lamb))[..., None]


    def normalize_across_scale(self, Stilde, w_S, zeta_S, pi_STinit, lamb0):
        self.set_normalization(Stilde, w_S * self.supportWeight(Stilde, lamb0), zeta_S, pi_STinit)

    def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb):
        return super().__call__(sSx, sSw * self.supportWeight(sSx, lamb ** 2), zeta_S, pi_ST)
