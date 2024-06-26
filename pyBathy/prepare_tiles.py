import numpy as np
from .utils import find_k_alpha_seed


def prepare_tiles(f, G, xy, cam, xm, ym, bathy):
    kL = 1.0
    maxNPix = bathy["params"]["maxNPix"]
    nf = bathy["params"]["nKeep"]
    lam1Norms = np.full(nf, np.nan)
    centerInds = np.full(nf, np.nan)
    Lx = bathy["params"]["Lx"]
    Ly = bathy["params"]["Ly"]

    idUse = np.where(
        (xy[:, 0] >= xm - Lx)
        & (xy[:, 0] <= xm + Lx)
        & (xy[:, 1] >= ym - Ly)
        & (xy[:, 1] <= ym + Ly)
    )[0]

    cams = cam[idUse]
    uniqueCams, N = np.unique(cams, return_counts=True)
    pick = np.argmax(N)
    camUsed = uniqueCams[pick]
    pick = np.where(cams == camUsed)[0]

    subG = G[:, idUse[pick]]
    subxy = xy[idUse[pick], :]

    validTile = False
    if len(idUse[pick]) >= 16:
        spanx = np.ptp(subxy[:, 0])
        spany = np.ptp(subxy[:, 1])
        if spanx >= 2 and spany >= 2:
            validTile = True

    if not validTile:
        fs = np.full(nf, np.nan)
        kAlpha0 = np.full((nf, 2), np.nan)
        subvs = []
        subXY = []
        return fs, kAlpha0, subvs, subXY, camUsed, lam1Norms, centerInds

    CAll = np.array(
        [
            np.matmul(subG[idx, :].T, subG[idx, :]) / len(idx)
            for idx in np.where(
                (f >= bathy["params"]["fB"][j] - (f[1] - f[0]) / 2)
                & (f <= bathy["params"]["fB"][j] + (f[1] - f[0]) / 2)
            )[0]
            for j in range(len(bathy["params"]["fB"]))
        ]
    )

    coh2 = np.sum(np.abs(CAll), axis=(1, 2))
    coh2Sortid = np.argsort(coh2)[::-1]
    fs = bathy["params"]["fB"][coh2Sortid[:nf]]

    kAlpha0 = np.full((len(fs), 2), np.nan)
    subvs = [None] * len(fs)
    subXY = [None] * len(fs)

    for i in range(len(fs)):
        indf = coh2Sortid[i]
        C = CAll[indf, :, :]
        eigvals, eigvecs = np.linalg.eigh(C)
        lam1Norms[i] = eigvals[-1]
        kAlpha0[i, :], centerInds[i] = find_k_alpha_seed(subxy, eigvecs[:, -1], xm, ym)

        LxTemp = np.pi / kAlpha0[i, 0] * kL
        LyTemp = LxTemp * Ly / Lx
        idUse = np.where(
            (subxy[:, 0] >= xm - LxTemp)
            & (subxy[:, 0] <= xm + LxTemp)
            & (subxy[:, 1] >= ym - LyTemp)
            & (subxy[:, 1] <= ym + LyTemp)
        )[0]

        if len(idUse) >= 16:
            subvs[i] = eigvecs[idUse, -1]
            subXY[i] = subxy[idUse, :]
            if len(idUse) > maxNPix:
                inds = np.linspace(0, len(subvs[i]) - 1, maxNPix, dtype=int)
                subvs[i] = subvs[i][inds]
                subXY[i] = subXY[i][inds, :]

            d = np.sqrt((subXY[i][:, 0] - xm) ** 2 + (subXY[i][:, 1] - ym) ** 2)
            centerInds[i] = np.argmin(d)
        else:
            fs[i] = np.nan
            subvs[i] = None
            subXY[i] = None
            lam1Norms[i] = np.nan

    return fs, kAlpha0, subvs, subXY, camUsed, lam1Norms, centerInds
