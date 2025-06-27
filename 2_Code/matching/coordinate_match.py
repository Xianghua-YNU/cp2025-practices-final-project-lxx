import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def find_matching_pairs(optical_dir, radio_dir, tol_arcsec=5):
    """通过坐标匹配光学和射电图像"""
    optical_files = [f for f in os.listdir(optical_dir) if f.endswith('.fits')]
    radio_files = [f for f in os.listdir(radio_dir) if f.endswith('.fits')]
    matches = []
    
    for o_file in optical_files:
        o_path = os.path.join(optical_dir, o_file)
        try:
            with fits.open(o_path) as o_hdu:
                o_wcs = WCS(o_hdu[0].header)
                o_ra, o_dec = o_wcs.wcs.crval
        except (OSError, AttributeError) as e:
            print(f"跳过无效的光学文件 {o_file}: {str(e)}")
            continue

        for r_file in radio_files:
            r_path = os.path.join(radio_dir, r_file)
            try:
                with fits.open(r_path) as r_hdu:
                    r_wcs = WCS(r_hdu[0].header)
                    r_ra, r_dec = r_wcs.wcs.crval
                
                dist = np.sqrt((o_ra - r_ra)**2 + (o_dec - r_dec)**2) * 3600
                if dist <= tol_arcsec:
                    matches.append((o_file, r_file, f"{dist:.2f}\""))
                    break
            except (OSError, AttributeError) as e:
                print(f"跳过无效的射电文件 {r_file}: {str(e)}")
                continue
    
    return matches
