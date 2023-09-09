from pykeops.torch import Vi, Vj


def getU(sigma, d, uCoeff):
    """
    Velocity field for the diffeomorphism

    Args:
        `sigma`: list of scales
        `d`: dimension of the ambient space (alway 3 even if effective dimension is 2)
        `uCoeff`: list of normalizing coefficients for each scale

    Return:
        `U`: is a KeOps function
    """
    if d == 3:
        print("d 3 in getU")
    xO, qyO, py, wpyO = Vi(0, d), Vj(1, d), Vj(2, d), Vj(3, 1)
    # retVal = xO.sqdist(qyO)*torch.tensor(0)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x, qy, wpy = xO / sig, qyO / sig, wpyO / sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp()  # G x N
        h = py + wpy * (x - qy)  # 1 X N x 3
        if sInd == 0:
            retVal = (1.0 / uCoeff[sInd]) * K * h
        else:
            retVal += (1.0 / uCoeff[sInd]) * K * h  # .sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1)  # G x 3


def getUdiv(sigma, d, uCoeff):
    """
    Divergence of the velocity field for the diffeomorphism

    Args:
        `sigma`: list of scales
        `d`: dimension of the ambient space (alway 3 even if effective dimension is 2)
        `uCoeff`: list of normalizing coefficients for each scale

    Return:
        `Udiv`: is a KeOps function
    """
    if d == 3:
        print("d 3 in uDiv")
    xO, qyO, py, wpyO = Vi(0, d), Vj(1, d), Vj(2, d), Vj(3, 1)
    for sInd in range(len(sigma)):
        sig = sigma[sInd]
        x, qy, wpy = xO / sig, qyO / sig, wpyO / sig
        D2 = x.sqdist(qy)
        K = (-D2 * 0.5).exp()
        h = wpy * (d - D2) - ((x - qy) * py).sum()
        if sInd == 0:
            retVal = (1.0 / (sig * uCoeff[sInd])) * K * h
        else:
            retVal += (1.0 / (sig * uCoeff[sInd])) * K * h  # .sum_reduction(axis=1)
    return retVal.sum_reduction(axis=1)  # G x 1
