from scipy.stats import t
import numpy as np

def bathy_ci(resid, jacob, w, flag):
    alpha = 0.05
    n = np.sum(w) / np.max(w)
    v = n - 1

    if flag == 1:
        _, R = np.linalg.qr(jacob, mode='complete')
        Rinv = np.linalg.inv(R)
        diagInfo = np.sum(Rinv**2, axis=1)

        rmse = np.linalg.norm(resid) / np.sqrt(v)
        se = np.sqrt(diagInfo) * rmse

        bathyErr = se * t.ppf(1 - alpha / 2, v)

    else:
        rmse = np.linalg.norm(resid) / np.sqrt(v)
        se = rmse * np.sqrt(np.diag(np.linalg.inv(jacob.T @ jacob)))
        bathyErr = se * t.ppf(1 - alpha / 2, v)

    return bathyErr