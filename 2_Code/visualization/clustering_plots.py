import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_cluster_distribution(labels, output_file='cluster_distribution.png'):
    """
    绘制聚类分布直方图
    Args:
        labels: 聚类标签数组
        output_file: 输出文件路径
    """
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Distribution')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    plt.close()

def plot_tsne(features, labels, output_file='tsne_visualization.png'):
    """
    使用t-SNE可视化高维特征
    Args:
        features: 特征矩阵
        labels: 聚类标签
        output_file: 输出文件路径
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_file)
    plt.close()
