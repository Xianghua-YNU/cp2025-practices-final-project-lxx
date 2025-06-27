import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def plot_and_save_matches(optical_dir, radio_dir, matches, output_dir='results/comparison_plots'):
    """绘制光学-射电匹配对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for optical_file, radio_file, offset in matches:
        try:
            # 读取光学图像
            o_path = os.path.join(optical_dir, optical_file)
            with fits.open(o_path) as o_hdu:
                o_data = o_hdu[0].data
                o_wcs = WCS(o_hdu[0].header)
                o_ra, o_dec = o_wcs.wcs.crval
                
            # 读取射电图像
            r_path = os.path.join(radio_dir, radio_file)
            with fits.open(r_path) as r_hdu:
                r_data = r_hdu[0].data
                r_wcs = WCS(r_hdu[0].header)
                r_ra, r_dec = r_wcs.wcs.crval
            
            # 创建对比图
            fig = plt.figure(figsize=(18, 8))
            
            # 光学图像
            ax1 = fig.add_subplot(121, projection=o_wcs)
            img1 = ax1.imshow(o_data, origin='lower', cmap='viridis', 
                            vmin=np.percentile(o_data, 5), 
                            vmax=np.percentile(o_data, 95))
            ax1.set_title(f'Optical: {optical_file}')
            ax1.grid(color='yellow', ls='--', alpha=0.3)
            
            # 射电图像
            ax2 = fig.add_subplot(122, projection=r_wcs)
            img2 = ax2.imshow(r_data, origin='lower', cmap='gray',
                            vmin=np.percentile(r_data, 5),
                            vmax=np.percentile(r_data, 95))
            ax2.set_title(f'Radio: {radio_file}\nOffset: {offset}')
            ax2.grid(color='cyan', ls='--', alpha=0.3)
            
            plt.tight_layout()
            plot_name = f"comparison_{optical_file.split('.')[0]}.png"
            plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight', dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {optical_file} and {radio_file}: {str(e)}")
            continue
