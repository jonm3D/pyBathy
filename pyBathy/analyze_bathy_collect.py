import numpy as np
from scipy.signal import detrend
from scipy.fftpack import fft
from .utils import find_interp_map, use_interp_map, plot_stacks_and_phase_maps
from .bathy_from_k_alpha import bathy_from_k_alpha
from .fix_bathy_tide import fix_bathy_tide
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .prep_bathy_input import prep_bathy_input

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

    # Find dominant frequencies for the entire collection region
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
    fs = f[GSortInd[:bathy["params"]["nKeep"]]]
    Gs = GBar[GSortInd[:bathy["params"]["nKeep"]]]

    bathy["fDependent"]["fB"] = np.tile(fs, (len(ym), len(xm), 1)).transpose((1, 2, 0))
    bathy["fDependent"]["lam1"] = np.tile(Gs, (len(ym), len(xm), 1)).transpose((1, 2, 0))

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

    print("Starting parallel processing for csm_invert_k_alpha...")

    def parallel_invert(args):
        yind, f, G, xyz, cam, x, y, bathy = args
        return csm_invert_k_alpha(f, G, xyz[:, :2], cam, x, y, bathy)

    with tqdm(total=len(bathy["xm"]) * len(bathy["ym"])) as pbar:
        for xind in range(len(bathy["xm"])):
            args = [(yind, f, G, xyz, cam, bathy["xm"][xind], bathy["ym"][yind], bathy) for yind in range(len(bathy["ym"]))]
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(parallel_invert, args)

            for yind, (fDep, camUsed) in enumerate(results):
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

                pbar.update(1)

    bathy = bathy_from_k_alpha(bathy)
    bathy = fix_bathy_tide(bathy)
    bathy["cpuTime"] = time.time() - start_time

    print("Completed parallel processing for csm_invert_k_alpha.")
    
    return bathy
