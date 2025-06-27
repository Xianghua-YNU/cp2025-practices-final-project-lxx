import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def plot_and_save_matches(optical_dir, radio_dir, matches, output_dir='comparison_plots'):
    """
    绘制光学-射电匹配对并保存图片
    Args:
        optical_dir: 光学FITS文件夹路径
        radio_dir: 射电FITS文件夹路径
        matches: 匹配对列表 [(光学文件, 射电文件, 偏移量), ...]
        output_dir: 输出图片文件夹
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for optical_file, radio_file, offset in matches:
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
        
        # 创建画布
        fig = plt.figure(figsize=(15, 6))
        
        # 绘制光学图像
        ax1 = fig.add_subplot(121, projection=o_wcs)
        img1 = ax1.imshow(o_data, origin='lower', cmap='viridis', 
                         vmin=np.percentile(o_data, 5), 
                         vmax=np.percentile(o_data, 95))
        ax1.set_title(f'Optical: {optical_file}\nRA={o_ra:.4f}°, DEC={o_dec:.4f}°')
        ax1.grid(color='yellow', ls='--', alpha=0.3)
        ax1.set_xlabel('RA (J2000)')
        ax1.set_ylabel('Dec (J2000)')
        
        # 绘制射电图像
        ax2 = fig.add_subplot(122, projection=r_wcs)
        img2 = ax2.imshow(r_data, origin='lower', cmap='gray',
                         vmin=np.percentile(r_data, 5),
                         vmax=np.percentile(r_data, 95))
        ax2.set_title(f'Radio: {radio_file}\nRA={r_ra:.4f}°, DEC={r_dec:.4f}°\nOffset: {offset}')
        ax2.grid(color='cyan', ls='--', alpha=0.3)
        ax2.set_xlabel('RA (J2000)')
        ax2.set_ylabel('Dec (J2000)')
        
        # 调整布局并保存
        plt.tight_layout()
        plot_name = f"comparison_RA{o_ra:.4f}_DEC{o_dec:.4f}.png"
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight', dpi=150)
        plt.close()
