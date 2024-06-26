import numpy as np

# Assuming these variables are defined elsewhere in your script or environment
global callCount, v, centerInd

def predict_csm(kAlpha, xyw):
    """
    Compute complex cross-spectral matrix from wavenumber and direction,
    kAlpha for x and y lags in first 2 cols of xyw. Third column accounts
    for weightings.

    Args:
        kAlpha: [k, alpha], wavenumber (2*pi/L) and direction (radians)
        xyw: array with delta_x, delta_y, weight

    Returns:
        q: complex correlation as a list of real and imaginary coefficients
    """
    global callCount, v, centerInd
    
    kx = -kAlpha[0] * np.cos(kAlpha[1])
    ky = -kAlpha[0] * np.sin(kAlpha[1])
    kxky = np.array([kx, ky])
    q = np.exp(1j * (np.dot(xyw[:, :2], kxky)))
    phi = np.angle(v[centerInd]) - np.angle(q[centerInd])
    q = q * np.exp(1j * phi)  # phase offset
    q = q * xyw[:, 2]
    
    q = np.concatenate([np.real(q), np.imag(q)])
    callCount += 1
    
    return q
