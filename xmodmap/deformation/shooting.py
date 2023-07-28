from xmodmap.deformation.hamiltonian import hamiltonianSystem, hamiltonianSystemGrid, hamiltonianSystemBackwards


def ralstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
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

    return f


def shooting(
    p0,
    q0,
    K0,
    sigma,
    d,
    numS,
    cA=1.0,
    cT=1.0,
    dimEff=3,
    single=False,
    nt=10,
    Integrator=ralstonIntegrator(),
):
    return Integrator(
        hamiltonianSystem(K0, sigma, d, numS, cA, cT, dimEff, single), (p0, q0), nt
    )


def ShootingGrid(
    p0,
    q0,
    qGrid,
    qGridw,
    K0,
    sigma,
    d,
    numS,
    uCoeff,
    cA=1.0,
    cT=1.0,
    dimEff=3,
    single=False,
    nt=10,
    Integrator=ralstonIntegrator(),
    T=None,
    wT=None,
):
    if T == None:
        return Integrator(
            hamiltonianSystemGrid(
                K0, sigma, d, numS, uCoeff, cA, cT, dimEff, single=single
            ),
            (p0[: (d + 1) * numS], q0, qGrid, qGridw),
            nt,
        )
    else:
        print("T shape adn wT shape")
        print(T.shape)
        print(wT.shape)
        print("G and wG shape")
        print(qGrid.shape)
        print(qGridw.shape)
        return Integrator(
            hamiltonianSystemGrid(
                K0, sigma, d, numS, uCoeff, cA, cT, dimEff, single=single
            ),
            (p0[: (d + 1) * numS], q0, qGrid, qGridw, T, wT),
            nt,
        )


def ShootingBackwards(
    p1,
    q1,
    T,
    wT,
    K0,
    sigma,
    d,
    numS,
    uCoeff,
    cA=1.0,
    cT=1.0,
    dimEff=3,
    single=False,
    nt=10,
    Integrator=ralstonIntegrator(),
):
    return Integrator(
        hamiltonianSystemBackwards(
            K0, sigma, d, numS, uCoeff, cA, cT, dimEff, single=single
        ),
        (-p1[: (d + 1) * numS], q1, T, wT),
        nt,
    )
