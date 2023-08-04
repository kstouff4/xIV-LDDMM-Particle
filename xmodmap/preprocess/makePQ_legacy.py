import torch

from xmodmap.preprocess import preprocess as preprocess


def makePQ(
    S,
    nu_S,
    T,
    nu_T,
    Csqpi=1.,
    lambInit=0.5,
    Csqlamb=1.,
    norm=True,
):
    '''
    Initialization of co-state (P) and state (Q) for Hamiltonian Control Optimization,
    transfer function between source and target feature spaces (Pi_ST),
    and support of target in target space (given by transition boundary defined through lambda parameter).

    Args:
        S = source positions (N x 3)
        nu_S = source feature values (N x labS)
        T = target positions (M x 3)
        nu_T = target feature values (M x labs)
        Csqpi = optimizing coefficient rescaling Pi_ST within optimization scheme
        lambInit = initial value of lambda (transition boundary width of target support)
        Csqlamb = optimizing coefficient rescaling lambda within optimization scheme
        norm = choice of initialization of Pi_ST
            * true uses uniform distribution over target feature values
            * false uses distribution over target feature values over whole target dataset

    Returns:
        w_S, w_T = total mass over features for each particle in S and T (N x 1, M x 1)
        zeta_S, zeta_T = (normalized) probability distribution over feature values for S and T (N x labS, M x labs)
        q0 = initial state (Stilde, w_S) values
        p0 = initial values of variables to be optimized, rescaled for optimization scheme (momenta (px,pw), Pi_ST, lambda)
        numS = number of particles in source
        Stilde, Ttilde = source and target positions rescaled within unit box
        s,m = scaling and translation applied to S and T to be within unit box
        pi_STinit = initial (user) value of transfer function Pi_ST
        lamb0 = initial (user) value of lambda

    '''
    # initialize state vectors based on normalization
    w_S = nu_S.sum(axis=-1)[..., None]
    w_T = nu_T.sum(axis=-1)[..., None]
    zeta_S = (nu_S / w_S)
    zeta_T = (nu_T / w_T)
    zeta_S[torch.squeeze(w_S == 0), ...] = torch.tensor(0.0)
    zeta_T[torch.squeeze(w_T == 0), ...] = torch.tensor(0.0)
    numS = w_S.shape[0]

    Stilde, Ttilde, s, m = preprocess.rescaleData(S, T)

    q0 = (
        torch.cat(
            (w_S.clone().detach().flatten(), Stilde.clone().detach().flatten()), 0
        )
        .requires_grad_(True)
    )  # not adding extra element for xc

    # two alternatives (linked to variation of KL divergence norm that consider)
    pi_STinit = torch.zeros((zeta_S.shape[-1], zeta_T.shape[-1]))
    nuSTtot = torch.sum(w_T) / torch.sum(w_S)
    if not norm:
        # feature distribution of target scaled by total mass in source
        pi_STinit[:, :] = nu_T.sum(axis=0) / torch.sum(w_S)
    else:
        pi_STinit[:, :] = torch.ones((1, nu_T.shape[-1])) / nu_T.shape[-1]
        pi_STinit = pi_STinit * nuSTtot

    print("pi shape ", pi_STinit.shape)
    print("initial Pi ", pi_STinit)
    print("unique values in Pi ", torch.unique(pi_STinit))

    lamb0 = lambInit
    if lambInit < 0:
        p0 = (
            torch.cat(
                (
                    torch.zeros_like(q0),
                    (1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach().flatten(),
                ),
                0,
            )
            .requires_grad_(True)
        )
    else:
        p0 = (
            torch.cat(
                (
                    torch.zeros_like(q0),
                    (1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach().flatten(),
                    (1.0 / Csqlamb) * torch.sqrt(lamb0).clone().detach().flatten(),
                ),
                0,
            )
            .requires_grad_(True)
        )

    return (
        w_S,
        w_T,
        zeta_S,
        zeta_T,
        q0,
        p0,
        numS,
        Stilde,
        Ttilde,
        s,
        m,
        pi_STinit,
        lamb0,
    )
