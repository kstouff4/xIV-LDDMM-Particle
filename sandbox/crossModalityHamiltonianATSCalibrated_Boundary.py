import os
import time

from matplotlib import pyplot as plt

from xmodmap.deformation.control.affine import getATauAlpha
from xmodmap.deformation.Shooting import Shooting, ShootingGrid, ShootingBackwards
from xmodmap.distances.boundary import supportRestrictionReg
from xmodmap.distances.kl import PiRegularizationSystem
from xmodmap.distances.varifold import LossVarifoldNorm
from xmodmap.model.CrossModality import LDDMMloss

import xmodmap.io.initialize as init
from sandbox.saveState import *
import xmodmap.io.getOutput as gO



##################################################################
# Print out Functions


def printCurrentVariables(
    Stilde,
    p0Curr,
    itCurr,
    sigmaRKHS,
    q0Curr,
    d,
    numS,
    zeta_S,
    labT,
    s,
    m,
    savedir,
    supportWeightF,
    cA=1.0,
    cS=10.0,
    cT=1.0,
    dimEff=3,
    single=False,
):
    p0 = p0Curr
    torch.save(p0, os.path.join(savedir, f"p0_iter{itCurr}.pt"))

    shoot = Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)
    pqList = shoot(p0Curr[: (d + 1) * numS], q0Curr)

    if supportWeightF is not None:
        pi_ST = p0Curr[(d + 1) * numS : -1].detach().view(zeta_S.shape[-1], labT)
        print("lambda: ", p0Curr[-1] ** 2)
    else:
        pi_ST = p0Curr[(d + 1) * numS :].detach().view(zeta_S.shape[-1], labT)

    pi_ST = pi_ST**2
    print("pi iter " + str(itCurr) + ", ", pi_ST)
    print(torch.unique(pi_ST))
    print("non diffeomorphism")
    totA = torch.zeros((3, 3))

    for i in range(len(pqList)):
        p = pqList[i][0]
        q = pqList[i][1]
        px = p[numS:].view(-1, d)
        pw = p[:numS].view(-1, 1)
        qx = q[numS:].view(-1, d)
        qw = q[:numS].view(-1, 1)  # torch.squeeze(q[:numS])[...,None]
        A, tau, Alpha = getATauAlpha(px, qx, pw, qw, dimEff=dimEff, single=single)
        totA += (0.1) * A  # assume 10 time steps?
        print("A, ", A.detach().cpu().numpy())
        print("tau, ", tau.detach().cpu().numpy())
        print("alpha, ", Alpha.detach().cpu().numpy())

    R = torch.linalg.matrix_exp(totA)
    print("R, ", R)
    print("det(R), ", torch.linalg.det(R))

    if pi_ST.shape[0] > pi_ST.shape[1] * 10:
        rat = torch.round(pi_ST.shape[0] / pi_ST.shape[1]).type(torch.IntTensor)
        pi_STplot = torch.zeros((pi_ST.shape[0], rat * pi_ST.shape[1]))
        for j in range(pi_ST.shape[1]):
            pi_STplot[:, j * rat : (j + 1) * rat] = torch.squeeze(pi_ST[:, j])[
                ..., None
            ]
    else:
        pi_STplot = pi_ST
    f, ax = plt.subplots()
    im = ax.imshow(pi_STplot.detach().cpu().numpy())
    f.colorbar(im, ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Label Replicates")
    f.savefig(os.path.join(savedir, f"pi_STiter{itCurr}.png"), dpi=300)

    q = pqList[-1][1]
    D = q[numS:].detach().view(-1, d)
    muD = q[0:numS].detach().view(-1, 1)
    nu_D = torch.squeeze(muD[0:numS])[..., None] * zeta_S
    zeta_D = nu_D / torch.sum(nu_D, axis=-1)[..., None]
    nu_Dpi = nu_D @ pi_ST
    Ds = init.resizeData(D, s, m)
    zeta_Dpi = nu_Dpi / torch.sum(nu_Dpi, axis=-1)[..., None]
    print("nu_D shape original: ", nu_D.shape)

    imageNamesSpi = ["weights", "maxImageVal"]
    imageNamesS = ["weights", "maxImageVal"]
    imageValsSpi = [torch.sum(nu_Dpi, axis=-1), torch.argmax(nu_Dpi, axis=-1)]
    imageValsS = [torch.sum(nu_D, axis=-1), torch.argmax(nu_D, axis=-1)]
    for i in range(zeta_Dpi.shape[-1]):
        imageValsSpi.append(zeta_Dpi[:, i])
        imageNamesSpi.append("zeta_" + str(i))
    for i in range(zeta_D.shape[-1]):
        imageValsS.append(zeta_D[:, i])
        imageNamesS.append("zeta_" + str(i))

    gO.writeVTK(
        Ds.cpu().numpy(),
        [i.cpu().numpy() for i in imageValsS],
        imageNamesS,
        os.path.join(savedir, f"testOutputiter{itCurr}_D10.vtk"),
        polyData=None,
    )
    gO.writeVTK(
        Ds.cpu().numpy(),
        [i.cpu().numpy() for i in imageValsSpi],
        imageNamesSpi,
        os.path.join(savedir, f"testOutputiter{itCurr}_Dpi10.vtk"),
        polyData=None,
    )

    if supportWeightF is not None:
        wS = supportWeightF(q[numS:].detach().view(-1, d), p0Curr[-1].detach() ** 2)
        nu_D = wS * torch.squeeze(muD[0:numS])[..., None] * zeta_S
        zeta_D = nu_D / torch.sum(nu_D, axis=-1)[..., None]
        nu_Dpi = nu_D @ pi_ST
        Ds = init.resizeData(D, s, m)
        zeta_Dpi = nu_Dpi / torch.sum(nu_Dpi, axis=-1)[..., None]
        imageValsSpi = [torch.sum(nu_Dpi, axis=-1), torch.argmax(nu_Dpi, axis=-1), wS]
        imageValsS = [torch.sum(nu_D, axis=-1), torch.argmax(nu_D, axis=-1), wS]
        imageNamesSpi = ["weights", "maxImageVal", "support_weights"]
        imageNamesS = ["weights", "maxImageVal", "support_weights"]
        for i in range(zeta_Dpi.shape[-1]):
            imageValsSpi.append(zeta_Dpi[:, i])
            imageNamesSpi.append("zeta_" + str(i))
        for i in range(zeta_D.shape[-1]):
            imageValsS.append(zeta_D[:, i])
            imageNamesS.append("zeta_" + str(i))

        gO.writeVTK(
            Ds.cpu().numpy(),
            [i.cpu().numpy() for i in imageValsS],
            imageNamesS,
            os.path.join(savedir, f"testOutputiter{itCurr}_D10Support.vtk"),
            polyData=None,
        )
        gO.writeVTK(
            Ds.cpu().numpy(),
            [i.cpu().numpy() for i in imageValsSpi],
            imageNamesSpi,
            os.path.join(savedir, f"testOutputiter{itCurr}_Dpi10Support.vtk"),
            polyData=None,
        )

    return pqList[-1][0], pqList[-1][1]


###################################################################
# Optimization


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

    Stilde, Ttilde, s, m = init.rescaleData(S, T)

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
    N = S.shape[0]

    pTilde = torch.zeros_like(p0)
    pTilde[0:numS] = torch.squeeze(
        torch.tensor(1.0 / (torch.sqrt(kScale) * dimEff * w_S))
    )  # torch.sqrt(kScale)*torch.sqrt(kScale)
    pTilde[numS : numS * (d + 1)] = torch.tensor(1.0 / torch.sqrt(kScale))  # torch.sqrt(kScale)*1.0/(cScale*torch.sqrt(kScale))
    if lamb0 < 0:
        pTilde[(d + 1) * numS :] = Csqpi
    else:
        pTilde[(d + 1) * numS : -1] = Csqpi
        pTilde[-1] = Csqlamb
    savepref = os.path.join(savedir, "State_")


    dataloss = LossVarifoldNorm(
        beta,
        sigmaVar,
        d,
        labs,
        w_S,
        w_T,
        zeta_S,
        zeta_T,
        pi_STinit,
        Stilde,
        Ttilde,
        lamb0
    )

    cst = dataloss.cst.detach()

    piLoss = PiRegularizationSystem(zeta_S, nu_T, numS, d)

    if lamb0 < 0:
        lambLoss = None
    else:
        lambLoss = supportRestrictionReg(eta0)

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

    optimizer = torch.optim.LBFGS(
        [p0],
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
        LH, LDA, LPI, LL = loss(p0 * pTilde, q0)
        L = LH + LDA + LPI + LL
        """
        print("loss", L.detach().cpu().numpy())
        print("loss H ", LH.detach().cpu().numpy())
        print("loss LDA ", LDA.detach().cpu().numpy())
        print("loss LPI ", LPI.detach().cpu().numpy())
        """

        # move the value to cpu() to be matplotlib compatible
        lossListH.append(LH.detach().clone().cpu())
        lossListDA.append(LDA.detach().clone().cpu())
        relLossList.append(LDA.detach().clone().cpu() / cst.clone().cpu())
        lossListPI.append(LPI.detach().clone().cpu())
        lossListL.append(LL.detach().clone().cpu())

        L.backward()
        return L

    def printCost(currIt):
        # save p0 for every 50th iteration and pi
        f, ax = plt.subplots()
        ax.plot(
            lossListH, label="H($q_0$,$p_0$), Final = {0:.6f}".format(lossListH[-1])
        )
        ax.plot(
            lossListDA, label="Varifold Norm, Final = {0:.6f}".format(lossListDA[-1])
        )
        ax.plot(
            torch.tensor(lossListDA).cpu() + torch.tensor(lossListH).cpu(),
            label="Total Cost, Final = {0:.6f}".format(
                lossListDA[-1] + lossListH[-1] + lossListPI[-1]
            ),
        )
        ax.plot(lossListPI, label="Pi Cost, Final = {0:.6f}".format(lossListPI[-1]))
        ax.plot(lossListL, label="Lambda Cost, Final = {0:.6f}".format(lossListL[-1]))
        ax.set_title("Loss")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.legend()
        f.savefig(os.path.join(savedir, f"Cost_{currIt}.png"), dpi=300)
        return

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

        saveState(osd, its, i, p0, savepref)
        if i > 0 and torch.isnan(lossListH[-1]) or torch.isnan(lossListDA[-1]):
            print("Exiting with detected NaN in Loss")
            print("state of optimizer")
            print(osd)
            break
        if i % 20 == 0:
            # p0Save = torch.clone(p0).detach()
            optimizer.zero_grad()
            p1, q1 = printCurrentVariables(
                Stilde,
                p0 * pTilde,
                i,
                sigmaRKHS,
                q0,
                d,
                numS,
                zeta_S,
                labs,
                s,
                m,
                savedir,
                dataloss.supportWeight,
                cA=cA,
                cS=cS,
                cT=cT,
                dimEff=dimEff,
                single=single,
            )
            printCost(i)

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
    saveVariables(q0, p0 * pTilde, Ttilde, w_T, s, m, savepref)

    printCost(its)

    f, ax = plt.subplots()
    ax.plot(lossOnlyH, label="TotLoss, Final = {0:.6f}".format(lossOnlyH[-1]))
    # ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA),label="Varifold Norm, Final = {0:.2f}".format(lossOnlyDA[-1]))
    # ax.plot(np.arange(len(lossOnlyH)),np.asarray(lossOnlyDA)+np.asarray(lossOnlyH),label="Total Cost, Final = {0:.2f}".format(lossOnlyDA[-1]+lossOnlyH[-1]))
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    f.savefig(os.path.join(savedir, "CostOuterIter.png"), dpi=300)

    # Print out deformed states
    # listpq = Shooting(p0, q0, Kg, Kv, sigmaRKHS,d,numS)
    coords = q0[numS:].detach().view(-1, d)
    rangesX = (torch.max(coords[..., 0]) - torch.min(coords[..., 0])) / 100.0
    rangesY = (torch.max(coords[..., 1]) - torch.min(coords[..., 1])) / 100.0
    rangesZ = (torch.max(coords[..., 2]) - torch.min(coords[..., 2])) / 100.0

    if rangesX == 0:
        rangesX = torch.max(coords[..., 0])
        xGrid = torch.arange(torch.min(coords[..., 0]) + 1.0)
    else:
        xGrid = torch.arange(
            torch.min(coords[..., 0]) - rangesX,
            torch.max(coords[..., 0]) + rangesX * 2,
            rangesX,
        )
    if rangesY == 0:
        rangesY = torch.max(coords[..., 1])
        yGrid = torch.arange(torch.min(coords[..., 1]) + 1.0)
    else:
        yGrid = torch.arange(
            torch.min(coords[..., 1]) - rangesY,
            torch.max(coords[..., 1]) + rangesY * 2,
            rangesY,
        )
    if rangesZ == 0:
        rangesZ = torch.max(coords[..., 2])
        zGrid = torch.arange(torch.min(coords[..., 2]) + 1.0)
    else:
        zGrid = torch.arange(
            torch.min(coords[..., 2]) - rangesZ,
            torch.max(coords[..., 2]) + rangesZ * 2,
            rangesZ,
        )

    XG, YG, ZG = torch.meshgrid((xGrid, yGrid, zGrid), indexing="ij")
    qGrid = torch.stack((XG.flatten(), YG.flatten(), ZG.flatten()), axis=-1)
    numG = qGrid.shape[0]
    qGrid = qGrid.flatten()
    qGridw = torch.ones((numG))

    shootgrid = ShootingGrid(sigmaRKHS, Stilde, cA=cA, cT=cT, dimEff=dimEff, single=single)
    listpq = shootgrid((p0 * pTilde)[: (d + 1) * numS], q0, qGrid, qGridw)

    print("length of pq list is, ", len(listpq))
    Dlist = []
    nu_Dlist = []
    nu_DPilist = []
    Glist = []
    wGlist = []
    Tlist = []
    nuTlist = []

    p0T = p0 * pTilde
    if lamb0 < 0:
        pi_STfinal = (
            p0T[(d + 1) * numS :].detach().view(zeta_S.shape[-1], zeta_T.shape[-1])
        )  # shouldn't matter multiplication by pTilde
    else:
        pi_STfinal = (
            p0T[(d + 1) * numS : -1].detach().view(zeta_S.shape[-1], zeta_T.shape[-1])
        )  # shouldn't matter multiplication by pTilde
    pi_STfinal = pi_STfinal**2
    print("pi final, ", pi_STfinal)
    print(torch.unique(pi_STfinal))

    if pi_STfinal.shape[0] > pi_STfinal.shape[1] * 10:
        rat = torch.round(pi_STfinal.shape[0] / pi_STfinal.shape[1]).type(
            torch.IntTensor
        )
        pi_STplot = torch.zeros((pi_STfinal.shape[0], rat * pi_STfinal.shape[1]))
        for j in range(pi_STfinal.shape[1]):
            pi_STplot[:, j * rat : (j + 1) * rat] = torch.squeeze(pi_STfinal[:, j])[
                ..., None
            ]
    else:
        pi_STplot = pi_STfinal

    f, ax = plt.subplots()
    im = ax.imshow(pi_STplot.cpu().numpy())
    f.colorbar(im, ax=ax)
    ax.set_ylabel("Source Labels")
    ax.set_xlabel("Target Label Replicates")
    f.savefig(os.path.join(savedir, "pi_STfinal.png"), dpi=300)

    for t in range(len(listpq)):
        qnp = listpq[t][1]
        D = qnp[numS:].detach().view(-1, d)
        muD = qnp.detach()
        nu_D = torch.squeeze(muD[0:numS])[..., None] * zeta_S.detach()
        nu_Dpi = nu_D @ pi_STfinal
        Dlist.append(init.resizeData(D, s, m))
        nu_Dlist.append(nu_D)
        nu_DPilist.append(nu_Dpi)

        gt = listpq[t][2]
        G = gt.detach().view(-1, d)
        Glist.append(init.resizeData(G, s, m))
        gw = listpq[t][3]
        wGlist.append(gw.detach())

    # Shoot Backwards
    # 7/01 = negative of momentum is incorporated into Shooting Backwards function
    shootBack = ShootingBackwards(sigmaRKHS, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)
    listBack = shootBack(listpq[-1][0][: (d + 1) * numS], listpq[-1][1], Ttilde.flatten(), w_T.flatten())

    for t in range(len(listBack)):
        Tt = listBack[t][2]
        Tlist.append(init.resizeData(Tt.detach().view(-1, d), s, m))
        wTt = listBack[t][3]
        nuTlist.append(wTt.detach()[..., None] * zeta_T.detach())

    # plot p0 as arrows
    listSp0 = torch.zeros((numS * 2, 3))
    polyListSp0 = torch.zeros((numS, 3))
    polyListSp0[:, 0] = 2
    polyListSp0[:, 1] = torch.arange(numS)  # +1
    polyListSp0[:, 2] = numS + torch.arange(numS)  # + 1

    listSp0[0:numS, :] = S.detach()
    listSp0[numS:, :] = (
        p0T[numS : (d + 1) * numS].detach().view(-1, d) + listSp0[0:numS, :]
    )
    featsp0 = torch.zeros((numS * 2, 1))
    featsp0[numS:, :] = p0T[0:numS].detach().view(-1, 1)
    gO.writeVTK(
        listSp0,
        [featsp0],
        ["p0_w"],
        os.path.join(savedir, "testOutput_p0.vtk"),
        polyData=polyListSp0,
    )

    pNew = pTilde * p0
    A, tau, Alpha = getATauAlpha(
        pNew[numS : (d + 1) * numS].view(-1, d),
        q0[numS:].view(-1, d),
        pNew[:numS].view(-1, 1),
        q0[:numS].view(-1, 1),
        dimEff=dimEff,
        single=single,
    )

    print("A final, ", A)
    print("tau final, ", tau)
    print("Alpha final, ", Alpha)

    A0 = A
    tau0 = tau
    p0 = pNew
    q0 = q0
    alpha0 = Alpha
    pTilde = pTilde
    pi_ST = pi_STfinal
    torch.save(
        [A0, tau0, p0, q0, alpha0, pTilde, pi_ST],
        os.path.join(savedir, "testOutput_values.pt"),
    )

    T = Ttilde
    torch.save([T, w_T, zeta_T, s, m], os.path.join(savedir, "testOutput_target.pt"))

    D = Dlist[-1]
    nu_D = nu_Dlist[-1]
    nu_Dpi = nu_DPilist[-1]
    Td = Tlist[-1]
    nu_Td = nuTlist[-1]
    torch.save(
        [D, nu_D, nu_Dpi, Td, nu_Td], os.path.join(savedir, "testOutput_Dvars.pt")
    )

    return Dlist, nu_Dlist, nu_DPilist, Glist, wGlist, Tlist, nuTlist
