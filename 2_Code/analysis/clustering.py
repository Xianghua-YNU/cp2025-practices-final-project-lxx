import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def cluster_features(features, n_clusters=5):
    """执行K-means聚类"""
    if len(features) < n_clusters:
        raise ValueError(f"Not enough samples ({len(features)}) for {n_clusters} clusters")
    
    print(f"\n正在进行K-means聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    with tqdm(total=100, desc="聚类进度") as pbar:
        labels = kmeans.fit_predict(features)
        pbar.update(100)
    return labels

def save_cluster_results(fits_files, labels, output_folder, input_folder):
    """保存聚类结果"""
    os.makedirs(output_folder, exist_ok=True)
    print("\n正在保存聚类结果:")
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        os.makedirs(os.path.join(output_folder, f'cluster_{label}'), exist_ok=True)
    
    for file, label in tqdm(zip(fits_files, labels), total=len(fits_files), desc="文件复制"):
        src_path = os.path.join(input_folder, file)
        dst_path = os.path.join(output_folder, f'cluster_{label}', file)
        shutil.copy(src_path, dst_path)
    
    with open(os.path.join(output_folder, 'labels.txt'), 'w') as f:
        for file, label in zip(fits_files, labels):
            f.write(f'{file}: {label}\n')
