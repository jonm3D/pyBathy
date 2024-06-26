
def localTransformPoints(origin, x_in, y_in, flag):
    import numpy as np
    """
    Transforms points either in geographical coordinates to local,
    or in local to geographical.
    This requires the local origin in geographical coordinates, as well as the
    angle between coordinate systems in CIRN angle convention.

    Args:
        origin (xo, yo, angle): local origin (0,0) in Geographical coordinates.
                Typically xo is E and yo is N coordinate.
                The angle should be the relative angle
                between the new (local) X axis  and old (Geo)
                X axis, positive counter-clockwise from the old (Geo) X.
        flag = 1 or 0 to indicate transform direction
              Geo-->local (1) or
              local-->Geo (0)
        x_in - Local (X) or Geo (E) coord depending on transform direction
        y_in = Local (Y) or Geo (N) coord depending on direction

    Returns:
        x_out - Local (X) or Geo (E) coord depending on direction
        y_out - Local (Y) or Geo (N) coord depending on direction

    """

    if flag == 1:
        # Geo to Local

        # Translate first
        easp = x_in - origin[0]
        norp = y_in - origin[1]

        # Rotate
        x_out = easp * np.cos(origin[2]) + norp * np.sin(origin[2])
        y_out = norp * np.cos(origin[2]) - easp * np.sin(origin[2])

    if flag == 0:
        # Local to Geo

        # Rotate first
        x_out = x_in * np.cos(origin[2]) - y_in * np.sin(origin[2])
        y_out = y_in * np.cos(origin[2]) + x_in * np.sin(origin[2])

        # Translate
        x_out = x_out + origin[0]
        y_out = y_out + origin[1]

    return x_out, y_out



def localTransformEquiGrid(origin, flag, x_in, y_in):
    import numpy as np
    """
    This function tranforms between Local World Coordinates and Geographical
    World Coordinates for equidistant grids. However, we cannot just rotate
    our local grid. The resolution would no longer be constant.So we find
    the local limits, rotate them in to world coordinates, and then make an
    equidistant grid. Local refers to the rotated coordinate system where X
    is positive offshore and y is oriented alongshore. The function can go
    from Local to Geographical and in reverse. Note, this only performs
    horizontal rotations/transformations. Function assumes transformed grid
    will have same square resolution as input grid.

    This requires the local origin in geographical coordinates, as well as the
    angle between coordinate systems in CIRN angle convention.

    Args:
        origin (xo, yo, angle): local origin (0,0) in Geographical coordinates.
                Typically xo is E and yo is N coordinate.
                The angle should be the relative angle
                between the new (local) X axis  and old (Geo)
                X axis, positive counter-clockwise from the old (Geo) X.
        flag = 1 or 0 to indicate transform direction
              Geo-->local (1) or
              local-->Geo (0)
        x_in - Local (XY) or Geo (EN) Grid depending on transformation
               direction. Should be equidistant in both X and Y and a valid
               meshgrid.
        y_in = Local (XY) or Geo (EN) Grid depending on transformation
               direction. Should be equidistant in both X and Y and a valid
               meshgrid.

    Returns:
        x_out - Local (XY) or Geo (EN) Grid depending on transformation
                direction. Should be equidistant in both X and Y and a valid
                meshgrid.

        y_out - Local (XY) or Geo (EN) Grid depending on transformation
                direction. Should be equidistant in both X and Y and a valid
                meshgrid.

    % translated from B. Bruder's localTransformEquiGrid.m matlab function, Apr 2024
    """

    # Section 1: Find Input Grid Extents + Resolution
    # Find Corners of XY Local Grid to Find Extents of AOI
    i_corners = np.array([
        [np.min(x_in), np.min(y_in)],  # [x, y]
        [np.min(x_in), np.max(y_in)],
        [np.max(x_in), np.max(y_in)],
        [np.max(x_in), np.min(y_in)]
    ])
    # Find Resolution, assuming dx and dy are equal.
    idxdy = np.nanmean(np.diff(x_in))
    # Difference dimension depends on how meshgrid created.
    if idxdy == 0:
        idxdy = np.nanmean(np.diff(np.transpose(x_in)))


    origin[2] = np.deg2rad(origin[2])

    # Section 2: Transform Input Grid Extents and Find Limits
    # Transform the Corners, depending on direction
    o_corners = localTransformPoints(origin, i_corners[:, 0], i_corners[:, 1], flag)
    # Find the limits of the AOI in Transformed Coordinates
    oxlim = [np.min(o_corners[0]), np.max(o_corners[0])]
    oylim = [np.min(o_corners[1]), np.max(o_corners[1])]
    #
    # Section 3: Create Equidistant Rotated Grid
    # Make Horizontal input Grid with same input resolution
    x_out, y_out = np.meshgrid(np.arange(oxlim[0], oxlim[1] + idxdy, idxdy),
                                   np.arange(oylim[0], oylim[1] + idxdy, idxdy))
    return x_out, y_out




def latlon2frf(lon,lat):

    import pyproj
    import numpy as np
    # EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("EPSG:3358")
    spE, spN = spNC(lon ,lat)

    r2d = 180.0 / np.pi;
    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d

    # to FRF coords
    spLengE = spE - Eom
    spLengN = spN - Nom
    R = np.sqrt(spLengE ** 2 + spLengN ** 2)
    Ang1 = np.arctan2(spLengE, spLengN)
    Ang2 = Ang1 + spAngle
    # to FRF
    X = R * np.sin(Ang2)
    Y = R * np.cos(Ang2)
    # to Lat Lon

    ans = {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN, 'X': X, 'Y': Y}
    return ans

def LatLon2ncsp(lon, lat):
    import pyproj
    """This function uses pyproj to convert longitude and latitude to stateplane

        test points taken from conversions made in USACE SMS modeling system

            nc stateplane  meters NAD83
            spE1 = 901926.2 m
            spN1 = 273871.0 m
            Lon1 = -75.75004989
            Lat1 =  36.17560399

            spE2 = 9025563.9 m
            spN2 = 276229.5 m
            lon2 = -75.47218285
            lat2 =  36.19666112

        Args:
        lon: geographic longitude (NAD83)  decimal degrees
        lat: geographic longitude (NAD83)  decimal degrees

        Returns:
        output dictionary with original coords and output of NC stateplane FIPS 3200
            'lat': latitude

            'lon': longitude

            'StateplaneE': NC stateplane

            'StateplaneN': NC stateplane

    """
    # EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("EPSG:3358")
    spE, spN = spNC(lon ,lat)
    ans = {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN}
    return ans



def ncsp2FRF(p1, p2):
    import numpy as np
    """this function converts nc StatePlane (3200 fips) to FRF coordinates
    based on kent Hathaways Code
    #
    #  15 Dec 2014
    #  Kent Hathaway.
    #  Translated from Matlab to python 2015-11-30 - Spicer Bak
    #
    #  Uses new fit (angles and scales) Bill Birkemeier determined in Nov 2014
    #
    #  This version will determine the input based on values, outputs FRF, lat/lon,
    #  and state plane coordinates.  Uses NAD83-2011.
    #
    #  IO:
    #  p1 = FRF X (m), or Longitude (deg + or -), or state plane Easting (m)
    #  p2 = FRF Y (m), or Latitude (deg), or state plane Northing (m)
    #
    #  X = FRF cross-shore (m)
    #  Y = FRF longshore (m)
    #  ALat = latitude (decimal degrees)
    #  ALon = longitude (decimal degrees, positive, or W)
    #  spN = state plane northing (m)
    #  spE = state plane easting (m)

    NAD83-86	2014
    Origin Latitude       36.1775975
    Origin Longitude      75.7496860
    m/degLat              110963.357
    m/degLon               89953.364
    GridAngle (deg)          18.1465
    Angle FRF to Lat/Lon     71.8535
    Angle FRF to State Grid  69.9747
    FRF Origin Northing  274093.1562
    Easting              901951.6805

    #  Debugging values
    p1=566.93;  p2=515.11;  % south rail at 1860
    ALat = 36.1836000
    ALon = 75.7454804
    p2= 36.18359977;
    p1=-75.74548109;
    SP:  p1 = 902307.92;
    p2 = 274771.22;

    Args:
      spE: North carolina state plane coordinate system - Easting
      spN: North carolina state plane coordinate system - Northing
      p1: first point
      p2: second point

    Returns:
      dictionary
       'xFRF': cross shore location in FRF coordinates

       'yFRF': alongshore location in FRF coodrindate system

       'StateplaneE': north carolina state plane coordinate system - easting

       'StateplaneN': north carolina state plane coordinate system - northing

    """
    r2d = 180.0 / np.pi;
    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d

    spE = p1
    spN = p2  # designating stateplane vars


    # to FRF coords
    spLengE = p1 - Eom
    spLengN = p2 - Nom
    R = np.sqrt(spLengE ** 2 + spLengN ** 2)
    Ang1 = np.arctan2(spLengE, spLengN)
    Ang2 = Ang1 + spAngle
    # to FRF
    X = R * np.sin(Ang2)
    Y = R * np.cos(Ang2)
    # to Lat Lon
    ans = {'xFRF': X,
           'yFRF': Y,
           'StateplaneE': spE,
           'StateplaneN': spN}
    return ans


def FRF2ncsp(xFRF, yFRF):
    import numpy as np
    """this function makes NC stateplane out of X and Y FRF coordinates,
    based on kent Hathaway's code, bill birkmeir's calculations .
    written by Kent Hathaway.      15 Dec 2014
    Translated from Matlab to python 2015-11-30 - Spicer Bak

    Uses new fit (angles and scales) Bill Birkemeier determined in Nov 2014
    This version will determine the input based on values, outputs FRF, lat/lon,
    and state plane coordinates.  Uses NAD83-2011.



    yFRF = FRF Y (m), or Latitude (deg), or state plane Northing (m)

    NAD83-86	2014
    Origin Latitude          36.1775975
    Origin Longitude         75.7496860
    m/degLat             110963.357
    m/degLon              89953.364
    GridAngle (deg)          18.1465
    Angle FRF to Lat/Lon     71.8535
    Angle FRF to State Grid  69.9747
    FRF Origin Northing  274093.1562
    Easting              901951.6805

    Test:

        xFRF=566.93;
        yFRF=515.11;  % south rail at 1860
        p1 = 902307.92;
        p2 = 274771.22;

    Args:
      xFRF: frf coordinate system cross-shore locatioin
      yFRF: frf coordinat system alongshore location

    Returns:
       xFRF: cross shore location in FRF coordinate system

       yFRF: alongshore location in FRF coodrinate system

       spE: North Carolina state plane coordinate system Easting

       spN: North Carolina State Plane coordinate system Northing

    """
    r2d = 180.0 / np.pi;

    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d
    X = xFRF
    Y = yFRF

    R = np.sqrt(X ** 2 + Y ** 2)
    Ang1 = np.arctan2(X, Y)  # % CW from Y
    #  to state plane
    Ang2 = Ang1 - spAngle
    AspN = R * np.cos(Ang2)
    AspE = R * np.sin(Ang2)
    spN = AspN + Nom
    spE = AspE + Eom
    out = {'xFRF': xFRF, 'yFRF': yFRF, 'StateplaneE': spE, 'StateplaneN': spN}
    return out



def ncsp2LatLon(spE, spN):
    import pyproj

    """This function uses pyproj to convert state plane to lat/lon

    test points taken from conversions made in USACE SMS modeling system

    nc stateplane  meters NAD83
    spE1 = 901926.2 m
    spN1 = 273871.0 m
    Lon1 = -75.75004989
    Lat1 =  36.17560399

    spE2 = 9025563.9 m
    spN2 = 276229.5 m
    lon2 = -75.47218285
    lat2 =  36.19666112

    Args:
      spE: easting - assumed north carolina state plane Meters
      spN: northing - assumed north carolina state plane meters

    Returns:
      dictionary with original coords and output of latitude and longitude.
        'lat': latitude
        'lon': longitude
        'StateplaneE': NC stateplane
        'StateplaneN': NC stateplane

    """
    #      from pyproj import CRS
    # >>> c1 = CRS(proj='latlong',datum='WGS84')
    # >>> x1 = -111.5; y1 = 45.25919444444
    # >>> c2 = CRS(proj="utm",zone=10,datum='NAD27')
    # >>> x2, y2 = transform(c1, c2, x1, y1)
    # >>> "%s  %s" % (str(x2)[:9],str(y2)[:9])
    # '1402291.0  5076289.5'

    EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("epsg:{}".format(EPSG))
    LL = pyproj.CRS(proj='latlon', datum='WGS84')  # pyproj.Proj('epsg:{}'.format(epsgLL))  # epsg for NAD83 projection
    lon, lat = pyproj.transform(spNC, LL, spE, spN)

    return {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN}





def FRF2latlon(xFRF, yFRF):
    import numpy as np
    """this function makes NC stateplane out of X and Y FRF coordinates,
    based on kent Hathaway's code, bill birkmeir's calculations .
    written by Kent Hathaway.      15 Dec 2014
    Translated from Matlab to python 2015-11-30 - Spicer Bak

    Uses new fit (angles and scales) Bill Birkemeier determined in Nov 2014
    This version will determine the input based on values, outputs FRF, lat/lon,
    and state plane coordinates.  Uses NAD83-2011.

    yFRF = FRF Y (m), or Latitude (deg), or state plane Northing (m)

    NAD83-86	2014
    Origin Latitude          36.1775975
    Origin Longitude         75.7496860
    m/degLat             110963.357
    m/degLon              89953.364
    GridAngle (deg)          18.1465
    Angle FRF to Lat/Lon     71.8535
    Angle FRF to State Grid  69.9747
    FRF Origin Northing  274093.1562
    Easting              901951.6805

    Test:

        xFRF=566.93;
        yFRF=515.11;  % south rail at 1860
        p1 = 902307.92;
        p2 = 274771.22;

    Args:
      xFRF: frf coordinate system cross-shore locatioin
      yFRF: frf coordinat system alongshore location

    Returns:
       xFRF: cross shore location in FRF coordinate system

       yFRF: alongshore location in FRF coodrinate system

       spE: North Carolina state plane coordinate system Easting

       spN: North Carolina State Plane coordinate system Northing

    """
    r2d = 180.0 / np.pi;

    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d
    X = xFRF
    Y = yFRF

    R = np.sqrt(X ** 2 + Y ** 2)
    Ang1 = np.arctan2(X, Y)  # % CW from Y
    #  to state plane
    Ang2 = Ang1 - spAngle
    AspN = R * np.cos(Ang2)
    AspE = R * np.sin(Ang2)
    spN = AspN + Nom
    spE = AspE + Eom

    # out = {'xFRF': xFRF, 'yFRF': yFRF, 'StateplaneE': spE, 'StateplaneN': spN}

    import pyproj

    """This function uses pyproj to convert state plane to lat/lon

    test points taken from conversions made in USACE SMS modeling system

    nc stateplane  meters NAD83
    spE1 = 901926.2 m
    spN1 = 273871.0 m
    Lon1 = -75.75004989
    Lat1 =  36.17560399

    spE2 = 9025563.9 m
    spN2 = 276229.5 m
    lon2 = -75.47218285
    lat2 =  36.19666112

    Args:
      spE: easting - assumed north carolina state plane Meters
      spN: northing - assumed north carolina state plane meters

    Returns:
      dictionary with original coords and output of latitude and longitude.
        'lat': latitude
        'lon': longitude
        'StateplaneE': NC stateplane
        'StateplaneN': NC stateplane

    """
    #      from pyproj import CRS
    # >>> c1 = CRS(proj='latlong',datum='WGS84')
    # >>> x1 = -111.5; y1 = 45.25919444444
    # >>> c2 = CRS(proj="utm",zone=10,datum='NAD27')
    # >>> x2, y2 = transform(c1, c2, x1, y1)
    # >>> "%s  %s" % (str(x2)[:9],str(y2)[:9])
    # '1402291.0  5076289.5'

    EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("epsg:{}".format(EPSG))
    LL = pyproj.CRS(proj='latlon', datum='WGS84')  # pyproj.Proj('epsg:{}'.format(epsgLL))  # epsg for NAD83 projection
    lon, lat = pyproj.transform(spNC, LL, spE, spN)

    ans = {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN, 'X': xFRF, 'Y': yFRF}
    return ans

    # return {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN}