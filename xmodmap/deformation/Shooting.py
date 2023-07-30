from xmodmap.deformation.Hamiltonian import HamiltonianSystem, HamiltonianSystemGrid

class Abstract_Shooting():
    HS = None

    def __init__(self, nt=10):
        self.nt = nt
        self.integrator = self.ralstonIntegrator

    def ralstonIntegrator(self, ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l



class Shooting(Abstract_Shooting):
    def __init__(self,  sigma, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=3, single=False, nt=10):
        super().__init__(nt=nt)

        self.sigma = sigma
        self.Stilde = Stilde
        self.cA = cA
        self.cS = cS
        self.cT = cT
        self.dimEff = dimEff
        self.single = single


        self.HS = HamiltonianSystem(self.sigma,
                                 self.Stilde,
                                 cA=self.cA,
                                 cS=self.cS,
                                 cT=self.cT,
                                 dimEff=self.dimEff,
                                 single=self.single)

    def __call__(self, p0, q0):
        return self.integrator(
            self.HS,
            (p0, q0),
            self.nt
        )


class ShootingGrid(Abstract_Shooting):
    def __init__(self,  sigma, Stilde, cA=1.0, cS=10.0,  cT=1.0, dimEff=3, single=False, nt=10):
        super().__init__(nt=nt)

        self.sigma = sigma
        self.Stilde = Stilde
        self.cA = cA
        self.cS = cS
        self.cT = cT
        self.dimEff = dimEff
        self.single = single


        self.HS = HamiltonianSystemGrid(self.sigma,
                                 self.Stilde,
                                 cA=self.cA,
                                 cS=self.cS,
                                 cT=self.cT,
                                 dimEff=self.dimEff,
                                 single=self.single)

    def __call__(self, p0, q0, qGrid, qGridw):
        return self.integrator(
            self.HS,
            (p0, q0, qGrid, qGridw),
            self.nt,
        )
