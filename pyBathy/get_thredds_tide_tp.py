import netCDF4 as nc
import datetime
import numpy as np


def getThreddsTideTp(t):
    """
    Get peak period and water level from the FRF thredds server

    Tries EOP first for water level, if doesn't work, uses 8m array

    Args:
        t: time in epoch (int)

    Returns:
        Tp: Peak period (float)
        WL: water level (float)
    """
    import datetime as dt
    from netCDF4 import Dataset
    import numpy as np

    time_obj = dt.datetime.utcfromtimestamp(int(t))
    hr = time_obj.hour
    yr = time_obj.year
    mon_str = str(str(time_obj.month).zfill(2))

    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"
    # Peak period Dataset
    ds = Dataset(
        frf_base
        + "oceanography/waves/8m-array/"
        + str(yr)
        + "/FRF-ocean_waves_8m-array_"
        + str(yr)
        + mon_str
        + ".nc",
        "r",
    )
    wave_Tp = ds.variables["waveTp"][:]
    thredds_time_Tp = np.asarray(ds.variables["time"][:])

    # Water Level Dataset
    try:
        # Try EOP
        ds2 = Dataset(
            frf_base
            + "oceanography/waterlevel/eopNoaaTide/"
            + str(yr)
            + "/FRF-ocean_waterlevel_eopNoaaTide_"
            + str(yr)
            + mon_str
            + ".nc",
            "r",
        )
        waterlevel = ds2.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds2.variables["time"][:])
        print("Water level sourced from EOP")
    except:
        # If no EOP, grab from 8m array
        waterlevel = ds.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds.variables["time"][:])
        print("Water level sourced from 8m array")

    ind_WL = np.abs(thredds_time_WL - t).argmin()
    ind_Tp = np.abs(thredds_time_Tp - t).argmin()

    # Peak period
    Tp = int(np.ceil(wave_Tp[ind_Tp]))

    # Water level
    WL = round(waterlevel[ind_WL], 2)

    return Tp, WL
