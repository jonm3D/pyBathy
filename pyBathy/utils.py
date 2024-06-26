import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt

def find_interp_map(xyz, pa, map=None, kn=1, do_nan=0):
    x = np.arange(pa[0], pa[2], pa[1])
    y = np.arange(pa[3], pa[5], pa[4])

    if map is not None:
        return x, y, map, None

    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    knn = NearestNeighbors(n_neighbors=kn)
    knn.fit(xyz[:, :2])
    distances, indices = knn.kneighbors(np.column_stack((X, Y)))

    wt = 1.0 / (distances + np.finfo(float).eps)
    wt = wt / np.sum(wt, axis=1, keepdims=True)

    if do_nan > 0:
        wt[distances > do_nan] = np.nan

    return x, y, indices, wt

def use_interp_map(I, map, wt):
    shape = map.shape
    interpolated_values = np.zeros(shape[0])

    for i in range(shape[0]):
        valid_indices = ~np.isnan(wt[i, :])
        if np.any(valid_indices):
            interpolated_values[i] = np.sum(I[map[i, valid_indices]] * wt[i, valid_indices])
        else:
            interpolated_values[i] = np.nan

    return interpolated_values

def plot_stacks_and_phase_maps(xyz, epoch, data, f, G, params):
    plt.figure(10)
    plt.clf()
    plt.scatter(xyz[:, 0], xyz[:, 1], c=np.angle(G[0, :]), s=1)
    plt.colorbar()
    plt.title('Phase Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.draw()

    plt.figure(11)
    plt.clf()
    plt.plot(epoch, np.mean(data, axis=1))
    plt.title('Time Stack')
    plt.xlabel('Time')
    plt.ylabel('Mean Intensity')
    plt.draw()

    plt.show()

def plot_region_of_interest(Xg, Yg, image, output_dir, collect_id):
    plt.figure()
    plt.pcolor(Xg, Yg, image, shading='flat', cmap='gray')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f'RegionOfInterest_{collect_id}.png'))
    plt.close()

def plot_results(Xg, Yg, image, bathy, f, G, xyz, output_dir, collect_id):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolor(Xg, Yg, image, shading='flat', cmap='gray')
    plt.subplot(1, 2, 2)
    plt.pcolor(bathy['xm'], bathy['ym'], -bathy['fCombined']['h'], shading='flat')
    plt.colorbar(label='Depth [m]')
    plt.savefig(os.path.join(output_dir, f'Bathy_{collect_id}.png'))
    plt.close()

    i = 4
    ind = np.argmin(np.abs(f - bathy['params']['fB'][i]))
    plt.figure()
    plt.scatter(xyz[:, 0], xyz[:, 1], c=np.angle(G[ind, :]), cmap='hsv', s=3)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar(label='Phase [rad]')
    plt.savefig(os.path.join(output_dir, f'PhaseMap_{collect_id}.png'))
    plt.close()

def k_invert_depth_model(h, f, w):
    g = 9.81  # gravity
    k = (2 * np.pi * f)**2 / (g * h)
    return k * w


def find_k_alpha_seed(subxy, v, xm, ym):
    k_alpha = [np.nan, np.nan]
    center_idx = np.nan

    if len(subxy) > 0:
        center_idx = np.argmin(np.sqrt((subxy[:, 0] - xm)**2 + (subxy[:, 1] - ym)**2))
        k_alpha = [1.0, 0.0]  # Replace with your actual logic to find k and alpha

    return k_alpha, center_idx