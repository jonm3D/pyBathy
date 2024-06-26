import numpy as np
from .utils import find_interp_map, use_interp_map
from .bathy_from_k_alpha_short import bathy_from_k_alpha_short
from .fix_bathy_tide import fix_bathy_tide
from .prepare_tiles import prepare_tiles
from .csm_invert_k_alpha_short import csm_invert_k_alpha_short
from scipy.signal import detrend
from scipy.fftpack import fft


def analyze_bathy_collect_short(xyz, epoch, data, cam, bathy, long_ok_flag=False):
    import time

    start_time = time.time()
    bathy["ver"] = "cBathyVersion"
    bathy["matVer"] = "Python"

    if not long_ok_flag:
        min_bands_for_v2 = 5
        not_short_record_length = (
            min_bands_for_v2
            / np.mean(np.diff(bathy["params"]["fB"]))
            / (epoch[1] - epoch[0])
        )
        if len(data) >= not_short_record_length:
            print(
                f"Warning - data record length {len(data)}s is longer than recommended."
            )
            print("It is recommended to NOT use cBathy short record analysis.")
            foo = input("Do you wish to continue? (type n or N to stop) ")
            if foo.lower() == "n":
                print("Aborting")
                return

    f, G, bathy = prep_bathy_input_short(xyz, epoch, data, bathy)

    if bathy["params"]["debug"]["DOPLOTSTACKANDPHASEMAPS"]:
        plot_stacks_and_phase_maps(xyz, epoch, data, f, G, bathy["params"])
        input("Hit return to continue ")
        plt.close(10)
        plt.close(11)

    data = data.astype(np.float64)
    IBar = np.mean(data, axis=0)
    IBright = np.max(data, axis=0)
    IDark = np.min(data, axis=0)
    xy = bathy["params"]["xyMinMax"]
    dxy = [bathy["params"]["dxm"], bathy["params"]["dym"]]
    pa = [xy[0], dxy[0], xy[1], xy[2], dxy[1], xy[3]]
    xm, ym, map, wt = find_interp_map(xyz, pa, [])
    timex = use_interp_map(IBar, map, wt)
    bathy["timex"] = timex.reshape((len(ym), len(xm)))
    bright = use_interp_map(IBright, map, wt)
    bathy["bright"] = bright.reshape((len(ym), len(xm)))
    dark = use_interp_map(IDark, map, wt)
    bathy["dark"] = dark.reshape((len(ym), len(xm)))

    timex_bar = np.mean(bathy["timex"], axis=0)
    min_ratio = 4
    min0 = np.min(timex_bar) - (np.max(timex_bar) - np.min(timex_bar)) / min_ratio
    weights = 1.0 / np.interp(xyz[:, 0], xm, timex_bar - min0)
    good = ~np.isnan(weights)
    GWt = G[:, good] * weights[good]
    GWt = GWt / np.sum(weights[good])
    GBar = np.mean(np.abs(GWt), axis=1)
    GBar2 = detrend(GBar)
    GSortInd = np.argsort(GBar)[::-1]
    fs = f[GSortInd[: bathy["params"]["nKeep"]]]
    Gs = GBar[GSortInd[: bathy["params"]["nKeep"]]]

    bathy["fDependent"]["fB"] = np.tile(fs, (len(ym), len(xm), 1))
    bathy["fDependent"]["lam1"] = np.tile(Gs, (len(ym), len(xm), 1))

    if bathy["params"]["debug"]["DOSHOWPROGRESS"]:
        plt.figure(21)
        plt.clf()
        plt.plot(xyz[:, 0], xyz[:, 1], ".")
        plt.axis("equal")
        plt.axis("tight")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Analysis Progress")
        plt.draw()
        plt.hold(True)

    for xind in range(len(bathy["xm"])):
        fDep = [None] * len(bathy["ym"])
        camUsed = np.zeros(len(bathy["ym"]))

        if bathy["params"]["debug"]["DOSHOWPROGRESS"]:
            for yind in range(len(bathy["ym"])):
                fDep[yind], camUsed[yind] = csm_invert_k_alpha_short(
                    f, G, xyz[:, :2], cam, bathy["xm"][xind], bathy["ym"][yind], bathy
                )
        else:
            for yind in range(len(bathy["ym"])):
                fDep[yind], camUsed[yind] = csm_invert_k_alpha_short(
                    f, G, xyz[:, :2], cam, bathy["xm"][xind], bathy["ym"][yind], bathy
                )

        for ind in range(len(bathy["ym"])):
            bathy["fDependent"]["kSeed"][ind, xind, :] = fDep[ind]["kSeed"]
            bathy["fDependent"]["aSeed"][ind, xind, :] = fDep[ind]["aSeed"]
            bathy["fDependent"]["camUsed"][ind, xind] = camUsed[ind]
            if any(~np.isnan(fDep[ind]["k"])):
                bathy["fDependent"]["k"][ind, xind, :] = fDep[ind]["k"]
                bathy["fDependent"]["a"][ind, xind, :] = fDep[ind]["a"]
                bathy["fDependent"]["dof"][ind, xind, :] = fDep[ind]["dof"]
                bathy["fDependent"]["skill"][ind, xind, :] = fDep[ind]["skill"]
                bathy["fDependent"]["lam1"][ind, xind, :] = fDep[ind]["lam1"]
                bathy["fDependent"]["kErr"][ind, xind, :] = fDep[ind]["kErr"]
                bathy["fDependent"]["aErr"][ind, xind, :] = fDep[ind]["aErr"]
                bathy["fDependent"]["hTemp"][ind, xind, :] = fDep[ind]["hTemp"]
                bathy["fDependent"]["hTempErr"][ind, xind, :] = fDep[ind]["hTempErr"]
                bathy["fDependent"]["NPixels"][ind, xind, :] = fDep[ind]["NPixels"]
                bathy["fDependent"]["NCalls"][ind, xind, :] = fDep[ind]["NCalls"]

    bathy = bathy_from_k_alpha_short(bathy)
    bathy = fix_bathy_tide(bathy)
    bathy["cpuTime"] = time.time() - start_time

    return bathy
