import numpy as np
from scipy.interpolate import griddata
from skimage.transform import radon

def find_k_alpha_seed(xy, v, xm, ym):
    """
    Given a phase tile v at locations xy, find good seed estimates for the
    wave angle and wavenumber that will be used for the nonlinear search in
    csmInvertKAlpha. This version uses the radon transform to find the
    dominant wave direction.
    Outputs are the two parameters, k and alpha, plus the center index
    of the pixel that is closest to xm, ym. This will be used to correct
    phase in predictCSM.
    """
    
    va = np.angle(v)

    dxy = 4
    x1, x2 = np.min(xy[:, 0]), np.max(xy[:, 0])
    y1, y2 = np.min(xy[:, 1]), np.max(xy[:, 1])
    x = np.arange(x1, x2 + dxy, dxy)
    y = np.arange(y1, y2 + dxy, dxy)
    X, Y = np.meshgrid(x, y)
    Iz0 = griddata((xy[:, 0], xy[:, 1]), va, (X, Y), method='linear')
    
    bad = np.isnan(Iz0)
    Iz = Iz0.copy()
    Iz[bad] = np.nanmean(Iz)
    
    theta = np.linspace(-np.pi / 4, np.pi / 4, 100)
    R = radon(Iz, theta * 180 / np.pi)
    
    c = np.argmax(np.var(R, axis=0))
    aSeed = -theta[c]
    
    corners = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
    xProj = corners[:, 0] * np.cos(aSeed) + corners[:, 1] * np.sin(aSeed)
    yProj = -corners[:, 0] * np.sin(aSeed) + corners[:, 1] * np.cos(aSeed)
    XProj, YProj = np.meshgrid(np.arange(np.min(xProj), np.max(xProj) + dxy, dxy),
                               np.arange(np.min(yProj), np.max(yProj) + dxy, dxy))
    XUnProj = XProj.flatten() * np.cos(aSeed) - YProj.flatten() * np.sin(aSeed)
    YUnProj = XProj.flatten() * np.sin(aSeed) + YProj.flatten() * np.cos(aSeed)
    IRot = griddata((X.flatten(), Y.flatten()), Iz0.flatten(), (XUnProj, YUnProj), method='linear')
    IRot = IRot.reshape(XProj.shape)
    
    dPhidx = np.diff(IRot, axis=1) / dxy
    kSeed = -np.nanmedian(dPhidx)
    
    d = np.sqrt((xy[:, 0] - xm) ** 2 + (xy[:, 1] - ym) ** 2)
    centerInd = np.argmin(d)
    
    kAlpha0 = np.array([kSeed, aSeed])
    
    return kAlpha0, centerInd