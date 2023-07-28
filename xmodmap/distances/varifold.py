def lossVarifoldNorm(T, w_T, zeta_T, zeta_S, K, d, numS, supportWeight):
    """
    K kernel for Varifold Norm (GaussLinKernel)
    """

    # print(w_T*zeta_T.cpu().numpy())
    cst = (K(T, T, w_T * zeta_T, w_T * zeta_T)).sum()

    def loss(sS, pi_est):
        # sS will be in the form of q (w_S,S,x_c)
        sSx = sS[numS:].view(-1, d)
        sSw = sS[:numS].view(-1, 1)
        if supportWeight is not None:
            wSupport = supportWeight(sSx, pi_est[-1] ** 2)
            sSw = wSupport * sSw
            pi_ST = (pi_est[:-1].view(zeta_S.shape[-1], zeta_T.shape[-1])) ** 2
        else:
            pi_ST = (pi_est.view(zeta_S.shape[-1], zeta_T.shape[-1])) ** 2
        nu_Spi = (sSw * zeta_S) @ pi_ST  # Ns x L * L x F

        k1 = K(sSx, sSx, nu_Spi, nu_Spi)
        k2 = K(sSx, T, nu_Spi, w_T * zeta_T)

        return (1.0 / 2.0) * (cst + k1.sum() - 2.0 * k2.sum())

    return cst.detach(), loss
