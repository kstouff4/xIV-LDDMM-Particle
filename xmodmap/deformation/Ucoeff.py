from pykeops.torch import Vi, Vj

class Ucoeff():
    """
    Compute the coefficients that basically scale the velocity field cost
    to be on the order of 1.0 for each scale.
    """
    def __init__(self, sigma, Stilde):
        self.N = Stilde.shape[0]
        self.d = Stilde.shape[1]

        self.sigma = sigma
        self.Stilde = Stilde

        self._uCoeff = []
        for sig in self.sigma:
            Kinit = self.GaussKernelSpaceSingle(sig)
            self._uCoeff.append(Kinit(self.Stilde, self.Stilde).sum() / (self.N * self.N * sig * sig))


    def GaussKernelSpaceSingle(self, sig):
        """
        k(x^\sigma - y^\sigma)
        """
        x, y = Vi(0, self.d) / sig, Vj(1, self.d) / sig
        D2 = x.sqdist(y)
        K = (-D2 * 0.5).exp()
        return K.sum_reduction(axis=1)

    @property
    def uCoeff(self):
        return self._uCoeff