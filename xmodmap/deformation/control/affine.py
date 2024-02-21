import torch

def getATauAlpha(px, qx, pw, qw, cA=1.0, cT=1.0, dimEff=3, single=False):
    """
    Compute Controls (A,Tau, Alpha) from momentums (px, qx, pw, qw)

    Args:
        `dimEff`: effective dimension of the data (warning, everything is hardcoded for 3D... even if effective dimension is 2)
        `qx`: is the source point cloud (N x 3)
        `px`: is the momentums (N x 3)
        `qw`: is the source features mass (N x 1) (total amount of mass across feature space)
        `pw`: is the feature mass momentums (N x 1)
        `single`: if True if homoscedastic, if False x,y are homoscadastic ans z coordinate is different

    Return:
        `A`: is skew symetric matrix (infinitesimal rotation)
        `tau`: is the (infinitesimal) translation vector
        `Alpha`: is a diagonal matrix
    """
    xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))  # moving barycenters; should be 1 x 3
    A = ((1.0 / (2.0 * cA)) * (px.T @ (qx - xc) - (qx - xc).T @ px))  # 3 x N * N x 3
    tau = ((1.0 / cT) * (px.sum(dim=0)))

    if qx.shape[-1] == 2:
        alpha = (px * (qx - xc)).sum() / 2. + (pw * qw).sum()
        Alpha = torch.diag(alpha.repeat(2))

    elif dimEff == 2:
        alpha = ((px[:,:2] * (qx[:,:2] - xc[:,:2])).sum() / 2. + (pw * qw).sum())
        Alpha = torch.diag(torch.cat((alpha.view(1), alpha.view(1), torch.zeros(1))))  # always scale Z by 0

    elif dimEff == 3 and single:
        alpha = ((px * (qx - xc)).sum() / 3. + (pw * qw).sum())
        Alpha = torch.diag(alpha.repeat(3))

    else:
        alpha_xy = ((px[:, 0:2] * (qx - xc)[:, 0:2]).sum() / 2. + (pw * qw).sum())
        alpha_z = ((px[:, -1] * (qx - xc)[:, -1]).sum() + (pw * qw).sum())
        Alpha = torch.diag(torch.cat((alpha_xy.view(1), alpha_xy.view(1), alpha_z.view(1))))

    return A, tau, Alpha
