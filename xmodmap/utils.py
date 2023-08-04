import torch


def createGrid(qx):

    # Print out deformed states
    coords = qx.detach()
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
    qGridw = torch.ones(numG, 1)

    return qGrid, qGridw