from scipy.optimize import least_squares
import numpy as np


def csm_invert_k_alpha(f, G, xy, cam, xm, ym, bathy):
    fs, kAlpha0, subvs, subXY, camUsed, lam1Norms, centerInds = prepare_tiles(
        f, G, xy, cam, xm, ym, bathy
    )

    fDep = {
        "kSeed": kAlpha0[:, 0],
        "aSeed": kAlpha0[:, 1],
        "k": np.nan * np.ones_like(kAlpha0[:, 0]),
        "a": np.nan * np.ones_like(kAlpha0[:, 0]),
        "dof": np.nan * np.ones_like(kAlpha0[:, 0]),
        "skill": np.nan * np.ones_like(kAlpha0[:, 0]),
        "lam1": lam1Norms,
        "kErr": np.nan * np.ones_like(kAlpha0[:, 0]),
        "aErr": np.nan * np.ones_like(kAlpha0[:, 0]),
        "hTemp": np.nan * np.ones_like(kAlpha0[:, 0]),
        "hTempErr": np.nan * np.ones_like(kAlpha0[:, 0]),
        "NPixels": np.nan * np.ones_like(kAlpha0[:, 0]),
        "NCalls": np.nan * np.ones_like(kAlpha0[:, 0]),
    }

    if any(np.isnan(kAlpha0[:, 0])):
        return fDep, camUsed

    for i in range(len(fs)):
        kAlphaPhi = [kAlpha0[i, 0], kAlpha0[i, 1], 0]
        result = least_squares(
            residuals, kAlphaPhi, args=(subXY[i], subvs[i]), method="lm"
        )

        if result.success:
            fDep["k"][i] = result.x[0]
            fDep["a"][i] = result.x[1]
            fDep["dof"][i] = np.sum(~np.isnan(subvs[i]))
            fDep["skill"][i] = np.sqrt(np.sum(result.fun**2))
            fDep["NPixels"][i] = len(subvs[i])
            fDep["NCalls"][i] = result.nfev

    return fDep, camUsed


def residuals(kAlpha, subXY, subvs):
    kx = -kAlpha[0] * np.cos(kAlpha[1])
    ky = -kAlpha[0] * np.sin(kAlpha[1])
    q = np.exp(1j * (subXY[:, 0] * kx + subXY[:, 1] * ky))
    phi = np.angle(subvs[0]) - np.angle(q[0])
    q *= np.exp(1j * phi)
    res = np.concatenate([np.real(q - subvs), np.imag(q - subvs)])
    return res
