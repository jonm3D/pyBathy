import numpy as np
from scipy.optimize import curve_fit
from .prepare_tiles import prepare_tiles
from .dispsol import dispsol
from .predict_csm import predict_csm

global callCount, v, centerInd


def csm_invert_k_alpha(f, G, xy, cam, xm, ym, bathy):
    g = 9.81
    ri = np.linspace(0, 1, 101)
    ai = (1 - np.cos(np.pi * (0.5 + 0.5 * ri))) ** 2

    fs, kAlpha0, subvs, subXY, camUsed, lam1Norms, centerInds = prepare_tiles(
        f, G, xy, cam, xm, ym, bathy
    )

    fDependent = {
        "fB": fs,
        "k": np.full_like(fs, np.nan),
        "a": np.full_like(fs, np.nan),
        "dof": np.full_like(fs, np.nan),
        "skill": np.full_like(fs, np.nan),
        "lam1": lam1Norms,
        "kErr": np.full_like(fs, np.nan),
        "aErr": np.full_like(fs, np.nan),
        "hTemp": np.full_like(fs, np.nan),
        "hTempErr": np.full_like(fs, np.nan),
        "NPixels": np.full_like(fs, np.nan),
        "NCalls": np.full_like(fs, np.nan),
        "kSeed": kAlpha0[:, 0],
        "aSeed": kAlpha0[:, 1],
    }

    for i in range(len(fs)):
        if np.isnan(fs[i]):
            continue

        hiimax = 9.8 * (1 / fs[i] ** 2) / (2 * np.pi) / 2
        hii = np.arange(bathy["params"]["MINDEPTH"], hiimax + 0.1, 0.1)
        kii, _ = dispsol(hii, fs[i])

        xy = subXY[i]
        Nxy = xy.shape[0]
        v = subvs[i]
        lam1Norm = lam1Norms[i]
        kAlphaInit = kAlpha0[i, :]

        dxmi = xy[:, 0] - xm
        dymi = xy[:, 1] - ym
        r = np.sqrt(
            (dxmi / bathy["params"]["Lx"]) ** 2 + (dymi / bathy["params"]["Ly"]) ** 2
        )
        Wmi = np.interp(r, ri, ai, left=0, right=0)
        w = np.abs(v) * Wmi

        try:
            kmin = (2 * np.pi * fs[i]) ** 2 / g
            kmax = 2 * np.pi * fs[i] / np.sqrt(g * bathy["params"]["MINDEPTH"])
            centerInd = centerInds[i]
            global callCount
            callCount = 0  # Reset call count

            popt, pcov = curve_fit(
                predict_csm,
                np.column_stack((xy, w)),
                np.concatenate([np.real(v), np.imag(v)]),
                p0=kAlphaInit,
                maxfev=2000,
            )
            kAlpha = popt

            if (
                kAlpha[0] < kmin
                or kAlpha[0] > kmax
                or kAlpha[1] > np.pi / 2
                or kAlpha[1] < -np.pi / 2
            ):
                raise ValueError("Resulting K, alpha are out of range")

            vPred = predict_csm(kAlpha, np.column_stack((xy, np.abs(v))))
            vPred = vPred[:Nxy] + 1j * vPred[Nxy:]
            skill = (
                1 - np.linalg.norm(vPred - v) ** 2 / np.linalg.norm(v - np.mean(v)) ** 2
            )

            ex = np.sqrt(np.diag(pcov))

        except Exception as e:
            kAlpha = [np.nan, np.nan]
            ex = kAlpha
            skill = np.nan
            lam1Norm = np.nan

        fDependent["k"][i] = kAlpha[0]
        fDependent["a"][i] = kAlpha[1]
        fDependent["dof"][i] = np.sum(w / (np.finfo(float).eps + np.max(w)))
        fDependent["skill"][i] = skill
        fDependent["lam1"][i] = lam1Norm
        fDependent["kErr"][i] = ex[0]
        fDependent["aErr"][i] = ex[1]

        if not np.isnan(kAlpha[0]):
            fDependent["hTemp"][i] = np.interp(kAlpha[0], kii, hii)
            dhiidkii = np.diff(hii) / np.diff(kii)
            fDependent["hTempErr"][i] = np.sqrt(
                (np.interp(kAlpha[0], kii[1:], dhiidkii) ** 2) * (ex[0] ** 2)
            )
        else:
            fDependent["hTemp"][i] = np.nan
            fDependent["hTempErr"][i] = np.nan

        fDependent["NPixels"][i] = len(v)
        fDependent["NCalls"][i] = callCount

    return fDependent, camUsed
