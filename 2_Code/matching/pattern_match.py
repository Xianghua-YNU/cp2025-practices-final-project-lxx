import os
import re

def match_by_pattern(optical_dir, radio_dir, pattern=r'(J\d{4}[+-]\d{4})'):
    """
    通过文件名模式匹配光学和射电文件
    Args:
        optical_dir: 光学FITS文件夹路径
        radio_dir: 射电FITS文件夹路径
        pattern: 用于匹配的正则表达式模式
    Returns:
        list: 匹配对列表 [(光学文件, 射电文件)]
    """
    optical_files = sorted([f for f in os.listdir(optical_dir) if f.endswith('.fits')])
    radio_files = sorted([f for f in os.listdir(radio_dir) if f.endswith('.fits')])
    
    matches = []
    for o_file in optical_files:
        o_match = re.search(pattern, o_file)
        if o_match:
            target_id = o_match.group(1)
            for r_file in radio_files:
                if target_id in r_file:
                    matches.append((o_file, r_file))
                    break
    return matches

def save_matches_to_csv(matches, output_file='pattern_matches.csv'):
    """将匹配结果保存到CSV文件"""
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optical File', 'Radio File'])
        writer.writerows(matches)
