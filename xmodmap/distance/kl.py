import torch


class PiRegularizationSystem:
    """
        pi regularization (KL divergence)
    """
    def __init__(self, zeta_S, nu_T, norm=True):
        self.nuTconst = torch.sum(nu_T)
        # two alternatives for distribution to compare to
        if not norm:
            self.nu_Tprob = torch.sum(nu_T, dim=0) / torch.sum(
                nu_T
            )  # compare to overall distribution of features
        else:
            self.nu_Tprob = (torch.ones((1, nu_T.shape[-1])) / nu_T.shape[-1])

        self.zeta_S = zeta_S
        self.nu_T = nu_T

        self.weight = 1.0 / torch.log(torch.tensor(nu_T.shape[-1])) if nu_T.shape[-1] > 1 else 1.0

    def __call__(self, qw, pi_est):

        mass_S = torch.sum(qw * self.zeta_S, dim=0)
        pi_ST = (pi_est.view(self.zeta_S.shape[-1], self.nu_T.shape[-1])) ** 2
        pi_S = torch.sum(pi_ST, dim=-1)
        pi_STprob = pi_ST / pi_S[..., None]
        numer = pi_STprob / self.nu_Tprob
        di = mass_S @ (pi_ST * (torch.log(numer)))
        return (1.0 / self.nuTconst) * di.sum()
