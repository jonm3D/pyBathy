import numpy as np
from scipy.optimize import least_squares
from .bathy_ci import bathy_ci
from .utils import k_invert_depth_model

def bathy_from_k_alpha_short(bathy):
    g = 9.81  # gravity
    ri = np.arange(0, 1.01, 0.01)
    ai = (1 - np.cos(np.pi * (0.5 + 0.5 * ri)))**2

    gammai = np.arange(0, 1.01, 0.01)
    gammaiBar = (gammai[:-1] + gammai[1:]) / 2
    foo = gammai * np.arctanh(gammai)
    dFoodGamma = np.diff(foo) / np.diff(gammai)
    mu = dFoodGamma / np.tanh(gammaiBar)

    params = bathy['params']
    nFreqs = bathy['fDependent']['fB'].shape[2]
    x = bathy['xm']
    y = bathy['ym']
    xMin = np.min(x)
    xMax = np.max(x)
    X, Y = np.meshgrid(x, y)

    for ix in range(len(x)):
        for iy in range(len(y)):
            kappa = 1 + (params['kappa0'] - 1) * (x[ix] - xMin) / (xMax - xMin)
            dxmi = X - x[ix]
            dymi = Y - y[iy]
            r = np.sqrt((dxmi / (params['Lx'] * kappa))**2 + (dymi / (params['Ly'] * kappa))**2)
            Wmi = np.interp(r, ri, ai, left=0, right=0)

            id = np.where((Wmi > 0) & (bathy['fDependent']['skill'] > params['QTOL']) & (~np.isnan(bathy['fDependent']['hTemp'])))
            if len(id[0]) >= params['minValsForBathyEst']:
                f = bathy['fDependent']['fB'][id]
                k = bathy['fDependent']['k'][id]
                kErr = bathy['fDependent']['kErr'][id]
                s = bathy['fDependent']['skill'][id]
                gamma = 4 * np.pi**2 * f**2 / (g * k)
                wMu = 1 / np.interp(gamma, gammaiBar, mu)
                w = Wmi[id] * wMu * s / (np.finfo(float).eps + k)
                hInit = np.sum(bathy['fDependent']['hTemp'][id] * s) / np.sum(s)

                result = least_squares(k_invert_depth_model, hInit, args=(f, k, w))

                if result.success:
                    h = result.x[0]
                    hErr = bathy_ci(result.fun, result.jac, w, 1)
                    kModel = k_invert_depth_model(h, f, w)
                    J = np.sqrt(np.sum(kModel * k * w) / (np.finfo(float).eps + np.sum(w**2)))
                    if (J != 0) and (h >= params['MINDEPTH']):
                        bathy['fCombined']['h'][iy, ix] = h
                        bathy['fCombined']['hErr'][iy, ix] = hErr
                        bathy['fCombined']['J'][iy, ix] = J
                        bathy['fCombined']['fBar'][iy, ix] = np.sum(f * w) / np.sum(w)

    return bathy