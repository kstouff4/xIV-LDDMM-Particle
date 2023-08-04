import os
import time

from xmodmap.distance.SupportRestrictionReg import supportRestrictionReg
from xmodmap.distance.kl import PiRegularizationSystem
from xmodmap.distance.LossVarifoldNorm import LossVarifoldNorm
from xmodmap.model.CrossModality import LDDMMloss

from sandbox.saveState import *
from xmodmap.preprocess.makePQ_legacy import makePQ


def callOptimize(
    S,
    nu_S,
    T,
    nu_T,
    sigmaRKHS,
    sigmaVar,
    gamma,
    d,
    labs,
    savedir,
    its=100,
    kScale=1.,
    cA=1.0,
    cT=1.0,
    cS=10.0,
    cPi=1.0,
    dimEff=3,
    Csqpi=1.,
    Csqlamb=1.,
    eta0=1.,
    lambInit=0.5,
    single=False,
    beta=None,
    loadPrevious=None,
):
    """
    Parameters:
        S, nu_S = source image varifold
        T, nu_T = target image varifold
        sigmaRKHS = list of sigmas for RKHS of velocity vector field (assumed Gaussian)
        sigmaVar = list of sigmas for varifold norm (assumed Gaussian)
        gammaA = weight on norm of infinitesimal rotation A (should be based on varifold norm in end decreasing to 0.01)
        gammaT = weight on norm of infinitesimal translation tau (should be based on varifold norm in end decreasing to 0.01)
        gammaU = weight on norm of velocity vector field (should be based on varifold norm in end decreasing to 0.01)
        d = dimensions of space
        labs = dimensions of feature space
        savedir = location to save cost graphs and p0 in
        its = iterations of LBFGS steps (each step has max 10 iterations and 15 evaluations of objective function)
        kScale = multiplicative factor of particles if applying parameter settings for one set of particles to another (larger), with assumption that mass has been conserved
        beta = computed to scale the varifold norm initially to 1; overridden with desired value if not None
        single = True if x,y,z should be same scaling
    """
    (
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
    ) = makePQ(S, nu_S, T, nu_T, Csqpi=Csqpi, lambInit=lambInit, Csqlamb=Csqlamb)

    qx = Stilde.clone().detach().requires_grad_(True)
    qw = w_S.clone().detach().requires_grad_(True)

    px = torch.zeros_like(qx).requires_grad_(True)
    pw = torch.zeros_like(qw).requires_grad_(True)

    pwTilde = torch.rsqrt(kScale) / dimEff / w_S
    pxTilde = torch.rsqrt(kScale)

    pi_ST = ((1.0 / Csqpi) * torch.sqrt(pi_STinit).clone().detach()).requires_grad_(True)

    if lamb0 < 0:
        variables_to_optimize = [px, pw, pi_ST]
        lambLoss = None
    else:
        lamb = (lamb0.sqrt().clone().detach() / Csqlamb).requires_grad_(True)
        variables_to_optimize = [px, pw, pi_ST, lamb]
        lambLoss = supportRestrictionReg(eta0)

    savepref = os.path.join(savedir, "State_")


    dataloss = LossVarifoldNorm(
        sigmaVar,
        w_T,
        zeta_T,
        Ttilde,
    )
    dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
    cst = dataloss.cst.detach()
    print("cst ", cst)

    piLoss = PiRegularizationSystem(zeta_S, nu_T, numS, d)


    loss = LDDMMloss(
        Stilde,
        sigmaRKHS,
        gamma,
        dataloss,
        piLoss,
        lambLoss,
        cA=cA,
        cS=cS,
        cT=cT,
        cPi=cPi,
        dimEff=dimEff,
        single=single,
    )

    '''
    saveParams(
        loss.hamiltonian.uCoeff,
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
    )
    
    saveParams([])
    
    '''

    #test_var = torch.cat((pw.flatten(),px.flatten(), pi_ST.flatten()), dim=0).detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        variables_to_optimize,
        #[test_var],
        max_eval=15,
        max_iter=10,
        line_search_fn="strong_wolfe",
        history_size=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )
    print("performing optimization...")
    start = time.time()

    # keep track of both losses
    lossListH = []
    lossListDA = []
    lossListPI = []
    lossListL = []
    relLossList = []
    lossOnlyH = []
    lossOnlyDA = []

    def closure():
        optimizer.zero_grad()

        LH, LDA, LPI, LL = loss(px * pxTilde,
                                pw * pwTilde,
                                qx,
                                qw,
                                pi_ST * Csqpi,
                                zeta_S)
        L = LH + LDA + LPI + LL

        # move the value to cpu() to be matplotlib compatible
        lossListH.append(LH.detach().clone().cpu())
        lossListDA.append(LDA.detach().clone().cpu())
        relLossList.append(LDA.detach().clone().cpu() / cst.clone().cpu())
        lossListPI.append(LPI.detach().clone().cpu())
        lossListL.append(LL.detach().clone().cpu())

        L.backward()
        return L


    if loadPrevious is not None:
        stateVars = loadState(loadPrevious)
        its = (
            stateVars["its"] + its
        )  # add the desired iterations on top of remaining previous ones
        p0 = stateVars["xopt"]
        optimizer.load_state_dict(stateVars["optimizer"])

    for i in range(its):
        print("it ", i, ": ", end="")
        optimizer.step(
            closure
        )  # default of 25 iterations in strong wolfe line search; will compute evals and iters until 25 unless reaches an optimum
        print("Current Losses", flush=True)
        print("H loss: ", lossListH[-1])
        print("Var loss: ", lossListDA[-1])
        print("Pi lossL ", lossListPI[-1])
        print("Lambda loss ", lossListL[-1])

        osd = optimizer.state_dict()
        lossOnlyH.append(osd["state"][0]["prev_loss"])

        #saveState(osd, its, i, p0, savepref)

        if i > 0 and torch.isnan(lossListH[-1]) or torch.isnan(lossListDA[-1]):
            print("Exiting with detected NaN in Loss")
            print("state of optimizer")
            print(osd)
            break
        if i % 20 == 0:
            optimizer.zero_grad()

            if i > 0:
                if torch.allclose(
                    lossListH[-1], lossListH[-2], atol=1e-6, rtol=1e-5
                ) and torch.allclose(
                    lossListDA[-1], lossListDA[-2], atol=1e-6, rtol=1e-5
                ):
                    print("state of optimizer")
                    print(osd)
                    break

    print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

    return Dlist, nu_Dlist, nu_DPilist, Glist, wGlist, Tlist, nuTlist
