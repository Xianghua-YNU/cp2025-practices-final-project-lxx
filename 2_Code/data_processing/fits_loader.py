import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def load_fits_files(fits_folder):
    """加载FITS文件数据"""
    fits_files = [f for f in os.listdir(fits_folder) if f.endswith('.fits')]
    fits_data_list = []
    print("\n正在加载FITS文件:")
    for file in tqdm(fits_files, desc="处理进度"):
        file_path = os.path.join(fits_folder, file)
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            fits_data_list.append(data)
    return np.array(fits_data_list), fits_files
