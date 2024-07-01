import numpy as np
from .find_k_alpha_seed import find_k_alpha_seed

def prepare_tiles(f, G, xy, cam, xm, ym, bathy):
    kL = 1.0
    max_n_pix = bathy['params']['maxNPix']
    n_f = bathy['params']['nKeep']
    lam1_norms = np.nan * np.ones(n_f)
    center_inds = np.nan * np.ones(n_f)
    Lx = bathy['params']['Lx']
    Ly = bathy['params']['Ly']
    fB = f

    id_use = np.where((xy[:, 0] >= xm - Lx) &
                      (xy[:, 0] <= xm + Lx) &
                      (xy[:, 1] >= ym - Ly) &
                      (xy[:, 1] <= ym + Ly))[0]

    cams = cam[id_use]
    unique_cams = np.unique(cams)
    N = np.array([np.sum(cams == uc) for uc in unique_cams])
    pick = []
    cam_used = -1
    if len(N) > 0:
        pick_cam = np.argmax(N)
        cam_used = unique_cams[pick_cam]
        pick = np.where(cams == cam_used)[0]

    subG = G[id_use[pick], :]
    subxy = xy[id_use[pick], :]

    min_n_pix = 16
    valid_tile = False
    if len(id_use[pick]) >= min_n_pix:
        min_spanx = 2
        min_spany = 2
        spanx = np.max(subxy[:, 0]) - np.min(subxy[:, 0])
        spany = np.max(subxy[:, 1]) - np.min(subxy[:, 1])
        if spanx >= min_spanx and spany >= min_spany:
            valid_tile = True

    if not valid_tile:
        nada = np.nan * np.ones(n_f)
        fs = nada
        kAlpha0 = np.nan * np.ones((n_f, 2))
        subvs = []
        subXY = []
        lam1_norms = nada
    else:
        fs = np.squeeze(bathy['fDependent']['fB'][0, 0, :])
        lam1_norms = np.squeeze(bathy['fDependent']['lam1'][0, 0, :])

        kAlpha0 = np.nan * np.ones((len(fs), 2))
        subvs = [None] * len(fs)
        subXY = [None] * len(fs)
        for i in range(len(fs)):
            indf = np.where(f == fs[i])[0]
            v = subG[:, indf].T
            kAlpha0[i, :], center_inds[i] = find_k_alpha_seed(subxy, v, xm, ym)

            Lx_temp = np.pi / kAlpha0[i, 0] * kL
            Ly_temp = Lx_temp * Ly / Lx
            id_use = np.where((subxy[:, 0] >= xm - Lx_temp) &
                              (subxy[:, 0] <= xm + Lx_temp) &
                              (subxy[:, 1] >= ym - Ly_temp) &
                              (subxy[:, 1] <= ym + Ly_temp))[0]
            valid_tile = False
            if len(id_use) >= min_n_pix:
                spanx = np.max(subxy[:, 0]) - np.min(subxy[:, 0])
                spany = np.max(subxy[:, 1]) - np.min(subxy[:, 1])
                if spanx >= min_spanx and spany >= min_spany:
                    valid_tile = True

            if valid_tile:
                subvs[i] = v[id_use]
                subXY[i] = subxy[id_use, :]
                del_step = max(1, len(id_use) // max_n_pix)
                if del_step > 1:
                    inds = np.round(np.arange(0, len(subvs[i]), del_step)).astype(int)
                    subvs[i] = subvs[i][inds]
                    subXY[i] = subXY[i][inds, :]
                    xy = subXY[i]
                    d = np.sqrt((xy[:, 0] - xm) ** 2 + (xy[:, 1] - ym) ** 2)
                    center_inds[i] = np.argmin(d)
            else:
                fs[i] = np.nan
                subvs[i] = []
                subXY[i] = []
                lam1_norms[i] = np.nan

    return fs, kAlpha0, subvs, subXY, cam_used, lam1_norms, center_inds