import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely import wkt
from frfFunctions import ncsp2FRF, LatLon2ncsp
import geopandas
import csv


name = []
dateTime = []
gsd = []
sat_az = []
sat_elev = []
x_sat_eci_km = []
y_sat_eci_km = []
z_sat_eci_km = []
qw_eci = []
qx_eci = []
qy_eci = []
qz_eci = []
x_sat_ecef_km = []
y_sat_ecef_km = []
z_sat_ecef_km = []
qw_ecef = []
qx_ecef = []
qy_ecef = []
qz_ecef = []
bit_depth = []
polygons = []
tifName = []
with open('/Users/dylananderson/Documents/projects/satelib/s108_20230511T190757Z/frame_index.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    next(csvreader)
    for row in csvreader:
        print('{}'.format(row[0]))
        name.append(row[0])
        dateTime.append(row[1])
        gsd.append(float(row[2]))
        sat_az.append(float(row[3]))
        sat_elev.append(float(row[4]))
        x_sat_eci_km.append(float(row[5]))
        y_sat_eci_km.append(float(row[6]))
        z_sat_eci_km.append(float(row[7]))
        qw_eci.append(float(row[8]))
        qx_eci.append(float(row[9]))
        qy_eci.append(float(row[10]))
        qz_eci.append(float(row[11]))
        x_sat_ecef_km.append(float(row[12]))
        y_sat_ecef_km.append(float(row[13]))
        z_sat_ecef_km.append(float(row[14]))
        qw_ecef.append(float(row[15]))
        qx_ecef.append(float(row[16]))
        qy_ecef.append(float(row[17]))
        qz_ecef.append(float(row[18]))
        bit_depth.append(float(row[19]))
        polygons.append(row[20])
        tifName.append(row[22])




skysat = geopandas.read_file('/Users/dylananderson/Documents/projects/satelib/s108_20230511T190757Z/frame_index.csv')
geodf = geopandas.GeoDataFrame(skysat, geometry=skysat.geom.apply(wkt.loads))#,crs="EPSG:4326")

coordinates = [list(geodf.geometry.exterior[row_id].coords) for row_id in range(geodf.shape[0])]
corner1 = [item[0] for item in coordinates]
corner2 = [item[1] for item in coordinates]
corner3 = [item[2] for item in coordinates]
corner4 = [item[3] for item in coordinates]

corner1x = np.asarray([item[0][0] for item in coordinates])
corner1y = np.asarray([item[0][1] for item in coordinates])
corner2x = np.asarray([item[1][0] for item in coordinates])
corner2y = np.asarray([item[1][1] for item in coordinates])
corner3x = np.asarray([item[2][0] for item in coordinates])
corner3y = np.asarray([item[2][1] for item in coordinates])
corner4x = np.asarray([item[3][0] for item in coordinates])
corner4y = np.asarray([item[3][1] for item in coordinates])

ncSPcorner1 = [LatLon2ncsp(corner1x[hh],corner1y[hh]) for hh in range(len(corner1x))]
frfCorner1 = [ncsp2FRF(ncSPcorner1[hh]['StateplaneE'],ncSPcorner1[hh]['StateplaneN']) for hh in range(len(corner1x))]
ncSPcorner2 = [LatLon2ncsp(corner2x[hh],corner2y[hh]) for hh in range(len(corner2x))]
frfCorner2 = [ncsp2FRF(ncSPcorner2[hh]['StateplaneE'],ncSPcorner2[hh]['StateplaneN']) for hh in range(len(corner2x))]
ncSPcorner3 = [LatLon2ncsp(corner3x[hh],corner3y[hh]) for hh in range(len(corner3x))]
frfCorner3 = [ncsp2FRF(ncSPcorner3[hh]['StateplaneE'],ncSPcorner3[hh]['StateplaneN']) for hh in range(len(corner3x))]
ncSPcorner4 = [LatLon2ncsp(corner4x[hh],corner4y[hh]) for hh in range(len(corner4x))]
frfCorner4 = [ncsp2FRF(ncSPcorner4[hh]['StateplaneE'],ncSPcorner4[hh]['StateplaneN']) for hh in range(len(corner4x))]

frfCorner1x = np.asarray([frfCorner1[hh]['xFRF'] for hh in range(len(corner1x))])
frfCorner1y = np.asarray([frfCorner1[hh]['yFRF'] for hh in range(len(corner1x))])
frfCorner2x = np.asarray([frfCorner2[hh]['xFRF'] for hh in range(len(corner1x))])
frfCorner2y = np.asarray([frfCorner2[hh]['yFRF'] for hh in range(len(corner1x))])
frfCorner3x = np.asarray([frfCorner3[hh]['xFRF'] for hh in range(len(corner1x))])
frfCorner3y = np.asarray([frfCorner3[hh]['yFRF'] for hh in range(len(corner1x))])
frfCorner4x = np.asarray([frfCorner4[hh]['xFRF'] for hh in range(len(corner1x))])
frfCorner4y = np.asarray([frfCorner4[hh]['yFRF'] for hh in range(len(corner1x))])



plt.figure()
plt.plot(frfCorner1x,frfCorner1y)
plt.plot(frfCorner2x,frfCorner2y)
plt.plot(frfCorner3x,frfCorner3y)
plt.plot(frfCorner4x,frfCorner4y)
plt.xlabel('xFRF (m)')
plt.ylabel('yFRF (m)')
plt.show()


azimuths = np.asarray(sat_az)
azimuths[np.where(azimuths<0)] = azimuths[np.where(azimuths<0)]+360
plt.figure()
p1 = plt.scatter(azimuths,sat_elev,c=gsd)
cb1 = plt.colorbar(p1)
plt.xlabel('azimuths')
plt.ylabel('elevations')
cb1.set_label('gsd (m)')
plt.show()



# rawVid = '/Users/dylananderson/Documents/projects/satelib/s112_20230905T145032Z/s112_20230905T145032Zstretched.avi'
rawVid = '/Users/dylananderson/Documents/projects/satelib/s111_20230605T191503Z/s111_20230605T191503Zstretched.avi'
# rawVid = '20230511_190757_ssc8d3__video.mp4'
# Read input video
cap = cv2.VideoCapture(rawVid)
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

imageStd = []
imageMean = []
# rawImage = np.ones((2000,2000))
for qq in range(n_frames):

    # Read next frame
    success, curr = cap.read()
    if not success:
        break
    # Convert to grayscale?
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # What do we want to know about the image?
    imageStd.append(np.std(curr))
    imageMean.append(np.mean(curr))
    # if qq == 0:
    #     rawImage2 = curr_gray
    # else:
    #     rawImage2 = np.dstack((rawImage2,curr_gray))


# plt.figure()
# p1 = plt.pcolor(np.flipud(np.std(rawImage,axis=2)),cmap='Greys_r')
# plt.show()



plt.figure()
p1 = plt.scatter(azimuths,sat_elev,c=imageMean)
cb1 = plt.colorbar(p1)
plt.xlabel('azimuths')
plt.ylabel('elevations')
cb1.set_label('image mean')
plt.show()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

PHI = 90-np.asarray(sat_elev)
THETA = azimuths

R = 500*np.ones(np.size(azimuths))
X = R * np.sin(PHI*np.pi/180) * np.cos(THETA*np.pi/180)
Y = R * np.sin(PHI*np.pi/180) * np.sin(THETA*np.pi/180)
Z = R * np.cos(PHI*np.pi/180)


# fig =plt.figure()
# ax1 = fig.add_subplot(projection='3d')
# p1 = ax1.scatter(X,Y,Z)
# p2 = ax1.scatter(0,0,0,'o')
# # cb1 = plt.colorbar(p1)
# # plt.xlabel('azimuths')
# # plt.ylabel('elevations')
# # cb1.set_label('image mean')
# set_axes_equal(ax1)
# plt.show()



from frfFunctions import FRF2latlon

sat1 = [FRF2latlon(100*X[hh],100*Y[hh]) for hh in range(len(X))]
sat1lat = np.asarray([sat1[hh]['lat'] for hh in range(len(X))])
sat1lon = np.asarray([sat1[hh]['lon'] for hh in range(len(X))])



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

from matplotlib.collections import PolyCollection

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Define lower left, uperright lontitude and lattitude respectively
extent = [-76.2, -75.2, 35.75, 36.75]
# Create a basemap instance that draws the Earth layer
bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
             urcrnrlon=extent[1], urcrnrlat=extent[3],
             projection='cyl', resolution='f', fix_aspect=False, ax=ax)
# Add Basemap to the figure
ax.add_collection3d(bm.drawcoastlines(linewidth=0.25))
ax.add_collection3d(bm.drawcountries(linewidth=0.35))
polys = []
for polygon in bm.landpolygons:
    polys.append(polygon.get_coords())
lc = PolyCollection(polys, edgecolor='black',
                    facecolor='#DDDDDD', closed=False)

ax.add_collection3d(lc)
ax.view_init(azim=-132, elev=29)
ax.set_xlabel('Longitude (°E)', labelpad=20)
ax.set_ylabel('Latitude (°N)', labelpad=20)
ax.set_zlabel('Altitude (km)', labelpad=20)
p1 = ax.scatter(sat1lon,sat1lat,Z,'.')

# Add meridian and parallel gridlines
lon_step = 0.5
lat_step = 0.5
meridians = np.arange(extent[0], extent[1] + lon_step, lon_step)
parallels = np.arange(extent[2], extent[3] + lat_step, lat_step)
ax.set_yticks(parallels)
ax.set_yticklabels(parallels)
ax.set_xticks(meridians)
ax.set_xticklabels(meridians)
ax.set_zlim(0., 550.)
plt.show()



