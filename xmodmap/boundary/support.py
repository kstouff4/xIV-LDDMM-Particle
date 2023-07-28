import torch


def defineSupport(Ttilde, eps=0.001):
    """
    This function defined the support of ROI as a zone of  the ambient space. It is actually a neighborhood of the
     *target* (defined by a sigmoid function, tanh). Coefficient alpha is though as 'transparency coefficient'.

    It involves the estimation of a normal vectors and points assuming parallel planes of data (coronal sections)

    Args:
        `Ttilde`: target location *scaled* in a unit box (N_target x 3)

    Return:
        `alphaSupportWeight
    """
    zMin = torch.min(Ttilde[:, -1])
    zMax = torch.max(Ttilde[:, -1])

    print("zMin: ", zMin)
    print("zMax: ", zMax)

    if zMin == zMax:
        zMin = torch.min(Ttilde[:, -2])
        zMax = torch.max(Ttilde[:, -2])
        print("new Zmin: ", zMin)
        print("new Zmax: ", zMax)
        sh = 0
        co = 1
        while sh < 2:
            lineZMin = Ttilde[
                torch.squeeze(
                    Ttilde[:, -2] < (zMin + torch.tensor(co * eps))
                ),
                ...,
            ]
            lineZMax = Ttilde[
                torch.squeeze(
                    Ttilde[:, -2] > (zMax - torch.tensor(co * eps))
                ),
                ...,
            ]
            sh = min(lineZMin.shape[0], lineZMax.shape[0])
            co += 1

        print("lineZMin: ", lineZMin)
        print("lineZMax: ", lineZMax)

        tCenter = torch.mean(Ttilde, axis=0)

        a0s = lineZMin[torch.randperm(lineZMin.shape[0])[0:2], ...]
        a1s = lineZMax[torch.randperm(lineZMax.shape[0])[0:2], ...]

        print("a0s: ", a0s)
        print("a1s: ", a1s)

        a0 = torch.mean(a0s, axis=0)
        a1 = torch.mean(a1s, axis=0)

        print("a0: ", a0)
        print("a1: ", a1)

        n0 = torch.tensor(
            [-(a0s[1, 1] - a0s[0, 1]), (a0s[1, 0] - a0s[0, 0]), Ttilde[0, -1]]
        )
        n1 = torch.tensor(
            [-(a1s[1, 1] - a1s[0, 1]), (a1s[1, 0] - a1s[0, 0]), Ttilde[0, -1]]
        )
        if torch.dot(tCenter - a0, n0) < 0:
            n0 = -1.0 * n0

        if torch.dot(tCenter - a1, n1) < 0:
            n1 = -1.0 * n1

    else:
        sh = 0
        co = 1
        while sh < 3:
            sliceZMin = Ttilde[
                torch.squeeze(
                    Ttilde[:, -1] < (zMin + torch.tensor(co * eps))
                ),
                ...,
            ]
            sliceZMax = Ttilde[
                torch.squeeze(
                    Ttilde[:, -1] > (zMax - torch.tensor(co * eps))
                ),
                ...,
            ]
            sh = min(sliceZMin.shape[0], sliceZMax.shape[0])
            co += 1

        print("sliceZMin: ", sliceZMin)
        print("sliceZMax: ", sliceZMax)

        tCenter = torch.mean(Ttilde, axis=0)

        # pick 3 points on each approximate slice and take center and normal vector
        a0s = sliceZMin[torch.randperm(sliceZMin.shape[0])[0:3], ...]
        a1s = sliceZMax[torch.randperm(sliceZMax.shape[0])[0:3], ...]

        print("a0s: ", a0s)
        print("a1s: ", a1s)

        a0 = torch.mean(a0s, axis=0)
        a1 = torch.mean(a1s, axis=0)

        n0 = torch.cross(a0s[1, ...] - a0s[0, ...], a0s[2, ...] - a0s[0, ...])
        if torch.dot(tCenter - a0, n0) < 0:
            n0 = -1.0 * n0

        n1 = torch.cross(a1s[1, ...] - a1s[0, ...], a1s[2, ...] - a1s[0, ...])
        if torch.dot(tCenter - a1, n1) < 0:
            n1 = -1.0 * n1

    # normalize vectors
    n0 = n0 / torch.sqrt(torch.sum(n0**2))
    n1 = n1 / torch.sqrt(torch.sum(n1**2))

    # ensure dot product with barycenter vector to point is positive, otherwise flip sign of normal
    print("n0: ", n0)
    print("n1: ", n1)

    def alphaSupportWeight(qx, lamb):
        return (
            0.5 * torch.tanh(torch.sum((qx - a0) * n0, axis=-1) / lamb)
            + 0.5 * torch.tanh(torch.sum((qx - a1) * n1, axis=-1) / lamb)
        )[..., None]

    return alphaSupportWeight
