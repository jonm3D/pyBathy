import numpy as np
from scipy.fftpack import fft
from scipy.signal import detrend

def prep_bathy_input(xyz, epoch, data, bathy):
    params = bathy["params"]

    # Ensure epoch is a column vector
    if epoch.ndim == 1 or epoch.shape[1] > epoch.shape[0]:
        epoch = epoch.reshape(-1, 1)

    fB = params["fB"]
    dfB = fB[1] - fB[0]

    # Detrend and FFT the data along the time axis (axis=1)
    data_detrended = detrend(data.astype(np.float64), axis=1)
    G = fft(data_detrended, axis=1)

    # Calculate the time step and frequency vector
    dt = np.mean(np.diff(epoch[:, 0]))
    df = 1 / (len(epoch) * dt)
    f = np.arange(0, 1 / (2 * dt), df)

    # Filter the frequency vector and corresponding FFT results
    freq_indices = np.where((f >= fB[0]) & (f <= fB[-1]))[0]
    f = f[freq_indices]
    G = G[:, freq_indices]

    # Size of x and y intervals
    dxm = params["dxm"]
    dym = params["dym"]

    # Span from the minimum X to maximum X in steps of dxm. Ditto Y.
    # Round lower boundary. If exists xyMinMax, let user set xm, ym.
    if params["xyMinMax"]:
        xm = np.arange(params["xyMinMax"][0], params["xyMinMax"][1] + dxm, dxm)
        ym = np.arange(params["xyMinMax"][2], params["xyMinMax"][3] + dym, dym)
    else:
        xm = np.arange(np.ceil(np.min(xyz[:, 0]) / dxm) * dxm, np.max(xyz[:, 0]) + dxm, dxm)
        ym = np.arange(np.ceil(np.min(xyz[:, 1]) / dym) * dym, np.max(xyz[:, 1]) + dym, dym)

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

# outputs something like
# f.shape
# (8,)
# G.shape
# (275000, 8)