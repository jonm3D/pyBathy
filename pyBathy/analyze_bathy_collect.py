import numpy as np
from scipy.signal import detrend
from scipy.fftpack import fft
from .utils import find_interp_map, use_interp_map, plot_stacks_and_phase_maps
from .bathy_from_k_alpha import bathy_from_k_alpha
from .csm_invert_k_alpha import csm_invert_k_alpha
from .fix_bathy_tide import fix_bathy_tide
from tqdm import tqdm
from .prep_bathy_input import prep_bathy_input
import matplotlib.pyplot as plt


def analyze_bathy_collect(xyz, epoch, data, cam, bathy):
    import time

    start_time = time.time()
    bathy["ver"] = "cBathyVersion"
    bathy["matVer"] = "Python"

    # Prepare data for analysis
    if epoch.ndim == 1:
        epoch = epoch.reshape(-1, 1)

    f, G, bathy = prep_bathy_input(xyz, epoch, data, bathy)

    if bathy["params"]["debug"]["DOPLOTSTACKANDPHASEMAPS"]:
        plot_stacks_and_phase_maps(xyz, epoch, data, f, G, bathy["params"])
        input("Hit return to continue ")
        plt.close(10)
        plt.close(11)

    # Create and save a time exposure, brightest, and darkest images
    data = data.astype(np.float64)
    IBar = np.mean(data, axis=1)
    IBright = np.max(data, axis=1)
    IDark = np.min(data, axis=1)
    xy = bathy["params"]["xyMinMax"]
    dxy = [bathy["params"]["dxm"], bathy["params"]["dym"]]
    pa = [xy[0], dxy[0], xy[1], xy[2], dxy[1], xy[3]]
    xm, ym, indices, weights = find_interp_map(xyz, pa, None)
    timex = use_interp_map(IBar, indices, weights)
    bathy["timex"] = timex.reshape((len(ym), len(xm)))
    bright = use_interp_map(IBright, indices, weights)
    bathy["bright"] = bright.reshape((len(ym), len(xm)))
    dark = use_interp_map(IDark, indices, weights)
    bathy["dark"] = dark.reshape((len(ym), len(xm)))

    # Find dominant frequencies for the entire collection region
    timex_bar = np.mean(bathy["timex"], axis=0)
    min_ratio = 4
    min0 = np.min(timex_bar) - (np.max(timex_bar) - np.min(timex_bar)) / min_ratio
    interp_weights = 1.0 / np.interp(xyz[:, 0], xm, timex_bar - min0)
    valid_weights = ~np.isnan(interp_weights)

    # Ensure valid_weights align with G's columns
    GWt = G[valid_weights, :] * interp_weights[valid_weights][:, np.newaxis]
    GWt = GWt / np.sum(interp_weights[valid_weights])
    GBar = np.mean(np.abs(GWt), axis=0)
    GBar2 = detrend(GBar)
    GSortInd = np.argsort(GBar)[::-1]
    fs = f[GSortInd[: bathy["params"]["nKeep"]]]
    Gs = GBar[GSortInd[: bathy["params"]["nKeep"]]]

    bathy["fDependent"]["fB"] = np.tile(fs, (len(ym), len(xm), 1)).transpose((1, 2, 0))
    bathy["fDependent"]["lam1"] = np.tile(Gs, (len(ym), len(xm), 1)).transpose(
        (1, 2, 0)
    )

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

    print("Starting processing for csm_invert_k_alpha...")

    with tqdm(total=len(bathy["xm"]) * len(bathy["ym"])) as progress_bar:
        for xind in range(len(bathy["xm"])):
            for yind in range(len(bathy["ym"])):
                fDep, camUsed = csm_invert_k_alpha(
                    f, G, xyz[:, :2], cam, bathy["xm"][xind], bathy["ym"][yind], bathy
                )

                bathy["fDependent"]["kSeed"][yind, xind, :] = fDep["kSeed"]
                bathy["fDependent"]["aSeed"][yind, xind, :] = fDep["aSeed"]
                bathy["fDependent"]["camUsed"][yind, xind] = camUsed

                if any(~np.isnan(fDep["k"])):
                    bathy["fDependent"]["k"][yind, xind, :] = fDep["k"]
                    bathy["fDependent"]["a"][yind, xind, :] = fDep["a"]
                    bathy["fDependent"]["dof"][yind, xind, :] = fDep["dof"]
                    bathy["fDependent"]["skill"][yind, xind, :] = fDep["skill"]
                    bathy["fDependent"]["lam1"][yind, xind, :] = fDep["lam1"]
                    bathy["fDependent"]["kErr"][yind, xind, :] = fDep["kErr"]
                    bathy["fDependent"]["aErr"][yind, xind, :] = fDep["aErr"]
                    bathy["fDependent"]["hTemp"][yind, xind, :] = fDep["hTemp"]
                    bathy["fDependent"]["hTempErr"][yind, xind, :] = fDep["hTempErr"]
                    bathy["fDependent"]["NPixels"][yind, xind, :] = fDep["NPixels"]
                    bathy["fDependent"]["NCalls"][yind, xind, :] = fDep["NCalls"]

                progress_bar.update(1)

    bathy = bathy_from_k_alpha(bathy)
    bathy = fix_bathy_tide(bathy)
    bathy["cpuTime"] = time.time() - start_time

    print("Completed processing for csm_invert_k_alpha.")

    return bathy
