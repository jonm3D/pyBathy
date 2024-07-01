import numpy as np
from scipy.optimize import minimize_scalar


def disp_eqn_k(k, h, f):
    g = 9.81
    omega = 2 * np.pi * f
    return np.abs(omega**2 - g * k * np.tanh(k * h))


def dispsol(h, f, flag=None):
    h = np.array(h).flatten()
    f = np.array(f).flatten()
    g = 9.81

    if flag is None:  # use iterative minimization technique
        k = []
        if len(h) == 1:
            for fi in f:
                res = minimize_scalar(
                    disp_eqn_k, bounds=(0, 20), args=(h[0], fi), method="bounded"
                )
                k.append(res.x)
        elif len(h) == len(f):
            for hi, fi in zip(h, f):
                res = minimize_scalar(
                    disp_eqn_k, bounds=(0, 20), args=(hi, fi), method="bounded"
                )
                k.append(res.x)
        else:
            for hi in h:
                res = minimize_scalar(
                    disp_eqn_k, bounds=(0, 20), args=(hi, f[0]), method="bounded"
                )
                k.append(res.x)
        k = np.array(k)
        kh = k * h
    else:
        if len(h) == 1:
            sigsq = (2 * np.pi * f) ** 2
            x = (sigsq * h) / g
        elif len(h) == len(f):
            sigsq = (2 * np.pi * f) ** 2
            x = (sigsq * h) / g
        else:
            sigsq = (2 * np.pi * f[0]) ** 2
            x = sigsq * h / g

        d1 = 0.6666666666
        d2 = 0.3555555555
        d3 = 0.1608465608
        d4 = 0.0632098765
        d5 = 0.0217540484
        d6 = 0.0065407983
        kh = np.sqrt(
            x**2
            + x
            / (1 + d1 * x + d2 * x**2 + d3 * x**3 + d4 * x**4 + d5 * x**5 + d6 * x**6)
        )
        if len(h) == 1:
            k = kh / h
        else:
            k = kh / h
    return k, kh
