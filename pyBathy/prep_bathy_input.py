import numpy as np
from scipy.fftpack import fft
from scipy.signal import detrend


def prep_bathy_input(xyz, epoch, data, bathy):
    params = bathy["params"]

    if epoch.shape[1] > epoch.shape[0]:
        epoch = epoch.T

    fB = params["fB"]
    dfB = fB[1] - fB[0]

    G = fft(detrend(data.astype(np.float64), axis=0), axis=0)

    dt = np.mean(np.diff(epoch, axis=0))
    df = 1 / (len(epoch) * dt)
    f = np.arange(0, 1 / (2 * dt), df)

    id = (f >= fB[0]) & (f <= fB[-1])
    f = f[id]
    G = G[id, :]

    dxm = params["dxm"]
    dym = params["dym"]

    if params["xyMinMax"]:
        xm = np.arange(params["xyMinMax"][0], params["xyMinMax"][1], dxm)
        ym = np.arange(params["xyMinMax"][2], params["xyMinMax"][3], dym)
    else:
        xm = np.arange(np.ceil(np.min(xyz[:, 0]) / dxm) * dxm, np.max(xyz[:, 0]), dxm)
        ym = np.arange(np.ceil(np.min(xyz[:, 1]) / dym) * dym, np.max(xyz[:, 1]), dym)

    bathy["tide"] = {"zt": np.nan, "e": 0, "source": ""}
    bathy["xm"] = xm
    bathy["ym"] = ym

    Nxm = len(xm)
    Nym = len(ym)
    nan_array = np.full((Nym, Nxm), np.nan)
    f_nan_array = np.full((Nym, Nxm, params["nKeep"]), np.nan)

    bathy["timex"] = nan_array.copy()
    bathy["bright"] = nan_array.copy()
    bathy["dark"] = nan_array.copy()
    bathy["fDependent"] = {
        "fB": f_nan_array.copy(),
        "k": f_nan_array.copy(),
        "a": f_nan_array.copy(),
        "hTemp": f_nan_array.copy(),
        "kErr": f_nan_array.copy(),
        "aErr": f_nan_array.copy(),
        "hTempErr": f_nan_array.copy(),
        "skill": f_nan_array.copy(),
        "dof": f_nan_array.copy(),
        "lam1": f_nan_array.copy(),
        "NPixels": f_nan_array.copy(),
        "NCalls": f_nan_array.copy(),
        "kSeed": f_nan_array.copy(),
        "aSeed": f_nan_array.copy(),
        "camUsed": nan_array.copy(),
    }
    bathy["fCombined"] = {
        "h": nan_array.copy(),
        "hErr": nan_array.copy(),
        "J": nan_array.copy(),
        "fBar": nan_array.copy(),
    }
    bathy["runningAverage"] = {
        "h": nan_array.copy(),
        "hErr": nan_array.copy(),
        "P": nan_array.copy(),
        "Q": nan_array.copy(),
    }
    bathy["cpuTime"] = np.nan

    return f, G, bathy
