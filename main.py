import os
import numpy as np
import re
from multiprocessing import Pool, cpu_count, Value, Lock
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import cv2  # OpenCV for faster image processing
from pyBathy.utils import plot_region_of_interest, plot_results
from pyBathy.skysat import skysat
from pyBathy.analyze_bathy_collect import analyze_bathy_collect
from pyBathy.prep_bathy_input import prep_bathy_input

# Global variables for shared counter and lock
counter = None
lock = None

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

def initialize_worker(shared_counter, shared_lock):
    global counter, lock
    counter = shared_counter
    lock = shared_lock

def load_and_preprocess_image(args):
    frames_dir, image_name, sy, ey, sx, ex, res, use_equalization = args
    image = cv2.imread(os.path.join(frames_dir, image_name), cv2.IMREAD_GRAYSCALE)
    image = np.flipud(image)
    result = cv2.equalizeHist(image[sy:ey:res, sx:ex:res]) if use_equalization else image[sy:ey:res, sx:ex:res]
    
    with lock:
        counter.value += 1
    
    return result

def process_directory(selected_dir, enable_plotting, use_equalization=False, max_seconds=None):
    print(f"Processing directory: {selected_dir}")
    frames_dir = os.path.join(selected_dir, 'frames')
    
    collect_id = re.search(r's\d{3}_\d{8}', selected_dir).group()
    date_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dc = [f for f in os.listdir(frames_dir) if f.endswith('.tiff')]
    if max_seconds is not None:
        max_frames = int(max_seconds * 30)  # Assuming 30 frames per second
        dc = dc[:max_frames]

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
    
    I = cv2.imread(os.path.join(frames_dir, dc[0]), cv2.IMREAD_GRAYSCALE)
    I = np.flipud(I)
    Ig = I[sy:ey:res, sx:ex:res]
    
    TS = np.zeros((Ig.shape[0], Ig.shape[1], len(range(0, len(dc), t_res))))

    # Shared counter for progress tracking
    shared_counter = Value('i', 0)
    shared_lock = Lock()

    print("Starting parallel processing for image loading and preprocessing...")
    with Pool(processes=cpu_count(), initializer=initialize_worker, initargs=(shared_counter, shared_lock)) as pool:
        tasks = [(frames_dir, dc[i], sy, ey, sx, ex, res, use_equalization) for i in range(0, len(dc), t_res)]
        results = list(tqdm(pool.imap(load_and_preprocess_image, tasks), total=len(tasks)))
        for ind, res in enumerate(results):
            TS[:, :, ind] = res
    print("Completed parallel processing for image loading and preprocessing.")
    
    if enable_plotting:
        print("Starting plotting of the region of interest...")
        plot_region_of_interest(Xg, Yg, TS[:, :, 0], output_dir, collect_id)
        print("Completed plotting of the region of interest.")
    
    tepoch_cut = t

    # Reshape data as in MATLAB version
    r, c = TS.shape[:2]
    xindgrid, yindgrid = np.meshgrid(np.arange(c), np.arange(r))
    rowIND = yindgrid.flatten()
    colIND = xindgrid.flatten()

    data = np.zeros((len(rowIND), TS.shape[2]))
    for i in range(len(rowIND)):
        data[i, :] = TS[rowIND[i], colIND[i], :]

    xyz = np.column_stack((Xg.flatten(), Yg.flatten(), np.zeros(Xg.size)))
    params = skysat()
    bathy = {'params': params}
    bathy['params']['xyMinMax'] = [0, 900, 0, 1000]
    cam = np.ones(xyz.shape[0])
    
    print("Preparing input for bathymetric analysis...")
    f, G, bathy = prep_bathy_input(xyz, tepoch_cut, data, bathy)
    print("Completed preparing input for bathymetric analysis. Running analyze_bathy_collect...")
    bathy = analyze_bathy_collect(xyz, tepoch_cut, data, cam, bathy)
    print("Completed analyze_bathy_collect. Generating plots and output...")
    
    if enable_plotting:
        print("Starting result plotting...")
        plot_results(Xg, Yg, TS[:, :, 0], bathy, f, G, xyz, output_dir, collect_id)
        print("Completed result plotting.")
    
    output_file_name = f'cBathy2Hz{int(t_tot)}s_{collect_id}.npz'
    np.savez_compressed(os.path.join(output_dir, output_file_name), bathy=bathy, data=TS, params=params, f=f, G=G, Xg=Xg, Yg=Yg, TS=TS, xyz=xyz, cam=cam, tepoch_cut=tepoch_cut)
    print(f"Saved results to {output_file_name}")

if __name__ == "__main__":
    base_dir = '/Users/jonathan/Desktop/frf_collects/'
    enable_plotting = False
    use_equalization = False  # Set default to False
    max_seconds = None  # Set the maximum number of seconds to load for faster testing
    
    directories = list_directories(base_dir)
    selected_idx = select_directory(directories)
    
    if selected_idx == len(directories) + 1:
        for dir_name in directories:
            process_directory(os.path.join(base_dir, dir_name), enable_plotting, use_equalization, max_seconds)
    else:
        process_directory(os.path.join(base_dir, directories[selected_idx - 1]), enable_plotting, use_equalization, max_seconds)
