import torch


def PiRegularizationSystem(zeta_S, nu_T, numS, d, norm=True):
    """
    pi regularization (KL divergence)
    """
    nuTconst = torch.sum(nu_T)
    # two alternatives for distribution to compare to
    if not norm:
        nu_Tprob = torch.sum(nu_T, dim=0) / torch.sum(
            nu_T
        )  # compare to overall distribution of features
    else:
        nu_Tprob = (torch.ones((1, nu_T.shape[-1])) / nu_T.shape[-1])

    def PiReg(qw, pi_est):
        mass_S = torch.sum(qw * zeta_S, dim=0)
        qwSum = torch.sum(qw, dim=0)[None, ...]
        pi_ST = (pi_est.view(zeta_S.shape[-1], nu_T.shape[-1])) ** 2
        pi_S = torch.sum(pi_ST, dim=-1)
        pi_STprob = pi_ST / pi_S[..., None]
        numer = pi_STprob / nu_Tprob
        di = mass_S @ (pi_ST * (torch.log(numer)))
        return (1.0 / nuTconst) * di.sum()

    return PiReg
