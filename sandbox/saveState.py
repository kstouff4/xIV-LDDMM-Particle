import torch


def saveState(osd, its, i, xopt, savepref):
    """
    osd = state of optimizer
    its = total iterations
    i = current iteration
    xopt = current optimization variable (p0*pTilde)
    """

    check_point = {"xopt": xopt, "its": its - i - 1, "optimizer": osd}
    filename = savepref + "_" + "checkpoint.pt"
    torch.save(check_point, filename)
    return


def loadState(filename):
    # Note: Input xopt & optimizer should be pre-defined.  This routine only updates their states.
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    if checkpoint["its"] == 0:
        checkpoint["its"] += 1
    print("new value of its", checkpoint["its"])
    return checkpoint


def saveParams(
    uCoeff,
    sigmaRKHS,
    sigmaVar,
    beta,
    d,
    labs,
    numS,
    pTilde,
    gamma,
    cA,
    cT,
    cPi,
    dimEff,
    single,
    savepref,
):
    """
    save parameters in dictionary
    """
    params = {
        "uCoeff": uCoeff,
        "sigmaRKHS": sigmaRKHS,
        "sigmaVar": sigmaVar,
        "beta": beta,
        "d": d,
        "labs": labs,
        "numS": numS,
        "pTilde": pTilde,
        "gamma": gamma,
        "cA": cA,
        "cT": cT,
        "cPi": cPi,
        "dimEff": dimEff,
        "single": single,
    }
    filename = savepref + "_" + "params.pt"
    torch.save(params, filename)
    return


def loadParams(filename):
    params = torch.load(filename)
    if not "single" in params:
        params["single"] = False
    if not "dimEff" in params:
        params["dimEff"] = 3
    return (
        params["uCoeff"],
        params["sigmaRKHS"],
        params["sigmaVar"],
        params["beta"],
        params["d"],
        params["labs"],
        params["numS"],
        params["pTilde"],
        params["gamma"],
        params["cA"],
        params["cT"],
        params["cPi"],
        params["dimEff"],
        params["single"],
    )


def saveVariables(q0, p0, Ttilde, wT, s, m, savepref):
    variables = {"q0": q0, "p0": p0, "Ttilde": Ttilde, "wT": wT, "s": s, "m": m}
    filename = savepref + "_" + "variables.pt"
    torch.save(variables, filename)
    return


def loadVariables(filename):
    variables = torch.load(filename)
    return (
        variables["q0"],
        variables["p0"],
        variables["Ttilde"],
        variables["wT"],
        variables["s"],
        variables["m"],
    )
