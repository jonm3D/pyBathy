import numpy as np

def skysat():
    params = {
        "stationStr": "skysat",
        "dxm": 10,
        "dym": 25,
        "xyMinMax": [80, 800, -500, 1500],
        "tideFunction": "get_thredds_tide_tp",
        "MINDEPTH": 0.25,
        "QTOL": 0.35,
        "minLam": 10,
        "Lx": 30,
        "Ly": 75,
        "kappa0": 2,
        "DECIMATE": 1,
        "maxNPix": 80,
        "minValsForBathyEst": 4,
        "shortLengthNFreqs": 4,
        "fB": np.arange(1 / 18, 1 / 4, 1 / 50),
        "nKeep": 4,
        "debug": {
            "production": 1,
            "DOPLOTSTACKANDPHASEMAPS": 1,
            "DOSHOWPROGRESS": 0,
            "DOPLOTPHASETILE": 0,
            "TRANSECTX": 200,
            "TRANSECTY": 900,
        },
        "offshoreRadCCWFromx": 0,
        "nlinfit": 1,
    }
    return params
