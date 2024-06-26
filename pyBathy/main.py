import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.io import imread
from skimage.exposure import equalize_adapthist
from scipy.signal import detrend
from scipy.fftpack import fft
from multiprocessing import Pool
from pyBathy import skysat, prep_bathy_input_short, analyze_bathy_collect_short

def list_directories(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and re.match(r's\d{3}_\d{8}', d)]

def select_directory(directories):
    print("Available directories:")
    for idx, dir_name in enumerate(directories):
        print(f"{idx + 1}: {dir_name}")
    print(f"{len(directories) + 1}: Process all directories")
    
    selected_idx = int(input("Select a directory by entering its number: "))
    if selected_idx < 1 or selected_idx > len(directories) + 1:
        raise ValueError("Invalid selection. Please restart the script and select a valid directory.")
    return selected_idx

def process_directory(selected_dir, enable_plotting):
    print(f"Processing directory: {selected_dir}")
    frames_dir = os.path.join(selected_dir, 'frames')
    
    collect_id = re.search(r's\d{3}_\d{8}', selected_dir).group()
    date_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dc = [f for f in os.listdir(frames_dir) if f.endswith('.tiff')]
    output_dir = os.path.join(selected_dir, f'output_{date_time_str}')
    os.makedirs(output_dir, exist_ok=True)
    
    Xv = np.arange(-700, 1300)
    Yv = np.arange(-300, 1700)
    X, Y = np.meshgrid(Xv, Yv)
    
    t_res = 1
    t_tot = len(dc) / 30
    t = np.arange(1/30, t_tot + 1/30, 1/30)
    
    res = 2
    sx, ex, sy, ey = 700, 1800, 300, 1300
    Xg, Yg = np.meshgrid(Xv[sx:ex:res], Yv[sy:ey:res])
    
    I = np.flipud(imread(os.path.join(frames_dir, dc[0])))
    Ig = I[sy:ey:res, sx:ex:res]
    
    TS = np.zeros((Ig.shape[0], Ig.shape[1], len(range(1, len(dc), t_res))))
    
    with Pool() as pool:
        TS = np.array(pool.starmap(load_and_preprocess_image, [(frames_dir, dc[i], sy, ey, sx, ex, res) for i in range(1, len(dc), t_res)]))
    
    if enable_plotting:
        plot_region_of_interest(Xg, Yg, TS[:, :, 0], output_dir, collect_id)
    
    tepoch_cut = t
    xyz = np.column_stack((Xg.flatten(), Yg.flatten(), np.zeros(Xg.size)))
    params = skysat()
    bathy = {'params': params}
    bathy['params']['xyMinMax'] = [0, 900, 0, 1000]
    cam = np.ones(xyz.shape[0])
    
    f, G, bathy = prep_bathy_input_short(xyz, tepoch_cut, TS, bathy)
    print('Done preparing input for analysis, running analyzeBathyCollect...')
    bathy = analyze_bathy_collect_short(xyz, tepoch_cut, TS, cam, bathy)
    print('Done analyzing data, generating plots and output...')
    
    if enable_plotting:
        plot_results(Xg, Yg, TS[:, :, 0], bathy, f, G, xyz, output_dir, collect_id)
    
    output_file_name = f'cBathy2Hz{int(t_tot)}s_{collect_id}.npz'
    np.savez_compressed(os.path.join(output_dir, output_file_name), bathy=bathy, data=TS, params=params, f=f, G=G, Xg=Xg, Yg=Yg, TS=TS, xyz=xyz, cam=cam, tepoch_cut=tepoch_cut)
    
def load_and_preprocess_image(frames_dir, image_name, sy, ey, sx, ex, res):
    image = np.flipud(imread(os.path.join(frames_dir, image_name)))
    return equalize_adapthist(image[sy:ey:res, sx:ex:res])

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

if __name__ == "__main__":
    base_dir = '/Users/jonathan/Desktop/frf_collects/'
    enable_plotting = False
    
    directories = list_directories(base_dir)
    selected_idx = select_directory(directories)
    
    if selected_idx == len(directories) + 1:
        for dir_name in directories:
            process_directory(os.path.join(base_dir, dir_name), enable_plotting)
    else:
        process_directory(os.path.join(base_dir, directories[selected_idx - 1]), enable_plotting)
