import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tqdm import tqdm

def evaluate_kmeans(features, max_k=10):
    """
    评估不同K值的聚类效果
    Args:
        features: 输入特征
        max_k: 最大K值
    Returns:
        dict: 包含各种评估指标的字典
    """
    print(f"\n正在评估K值 (2-{max_k}):")
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    db_scores = []
    k_values = range(2, max_k+1)
    
    for k in tqdm(k_values, desc="K值评估"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))
        calinski_scores.append(calinski_harabasz_score(features, labels))
        db_scores.append(davies_bouldin_score(features, labels))
    
    # 绘制评估图表
    _plot_evaluation_results(k_values, inertias, silhouette_scores, calinski_scores, db_scores)
    
    return {
        'inertia': inertias,
        'silhouette': silhouette_scores,
        'calinski': calinski_scores,
        'davies_bouldin': db_scores
    }

def _plot_evaluation_results(k_values, inertias, silhouette_scores, calinski_scores, db_scores):
    """内部函数：绘制评估结果图表"""
    # Inertia (Elbow Method)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig('elbow_method.png')
    plt.close()

    # Silhouette Score
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (Higher is better)')
    plt.grid(True)
    plt.savefig('silhouette_score.png')
    plt.close()

    # Calinski-Harabasz Score
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, calinski_scores, 'go-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Index (Higher is better)')
    plt.grid(True)
    plt.savefig('calinski_harabasz.png')
    plt.close()

    # Davies-Bouldin Score
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, db_scores, 'mo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Index (Lower is better)')
    plt.grid(True)
    plt.savefig('davies_bouldin.png')
    plt.close()
