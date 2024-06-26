def fix_bathy_tide(bathy):
    try:
        tide_function = eval(bathy["params"]["tideFunction"])
        tide = tide_function(bathy["sName"], bathy["epoch"])
        if not bathy.get("tide"):
            bathy["tide"] = {"zt": np.nan}
        tidecorr = np.sum([bathy["tide"]["zt"], -tide["zt"]])
        bathy["fDependent"]["hTemp"] += tidecorr
        bathy["fCombined"]["h"] += tidecorr
        bathy["tide"] = tide
    except:
        pass
    return bathy
