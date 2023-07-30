from pykeops.torch import Vi, Vj

class Ucoeff():
    def __init__(self, sigma, Stilde, cS):
        # TODO: avoid to pack and unpack p and q
        self.d = Stilde.shape[1]
        self.numS = Stilde.shape[0]

        self.sigma = sigma
        self.cS = cS
        self.Stilde = Stilde

        self.N = Stilde.shape[0]

        self._uCoeff = []
        for sig in self.sigma:
            Kinit = self.GaussKernelSpaceSingle(sig)
            self._uCoeff.append(self.cS * Kinit(self.Stilde, self.Stilde).sum() / (self.N * self.N * sig * sig))

        # for ss in range(len(self._uCoeff)):
        #     print("sig is ", self.sigma[ss])
        #     print("uCoeff ", self._uCoeff[ss])

        # print("here !")

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