import os
import torch
import timm
import numpy as np
from tqdm import tqdm
from data_processing.fits_loader import load_fits_files
from data_processing.image_preprocessor import preprocess_image
from data_processing.feature_extractor import extract_features
from analysis.clustering import cluster_features, save_cluster_results
from analysis.evaluation import evaluate_kmeans
from matching.coordinate_match import find_matching_pairs
from visualization.comparison_plots import plot_and_save_matches
from visualization.clustering_plots import plot_cluster_distribution, plot_tsne

def main():
    # 初始化设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_folder = 'data/optical'  # 替换为实际路径
    output_folder = 'results/clusters'
    radio_dir = "data/radio"  # 替换为实际路径
    
    # 1. 加载模型
    print("\n正在加载DenseNet201模型...")
    model = timm.create_model('densenet201', pretrained=False)
    model.load_state_dict(torch.load('models/densenet201.pth'), strict=False)
    model = model.to(device)
    
    # 2. 数据处理流程
    fits_data, fits_files = load_fits_files(input_folder)
    preprocessed_images = [preprocess_image(data) for data in tqdm(fits_data, desc="图像预处理")]
    features = extract_features(model, preprocessed_images, device)
    
    # 3. 聚类分析
    evaluation_results = evaluate_kmeans(features, max_k=10)
    best_k = np.argmax(evaluation_results['silhouette']) + 2
    print(f"\n推荐聚类数量: {best_k}")
    
    labels = cluster_features(features, n_clusters=best_k)
    save_cluster_results(fits_files, labels, output_folder, input_folder)
    
    # 可视化聚类结果
    plot_cluster_distribution(labels)
    plot_tsne(features, labels)
    
    # 4. 光学-射电匹配
    coord_matches = find_matching_pairs(output_folder, radio_dir)
    plot_and_save_matches(output_folder, radio_dir, coord_matches)

if __name__ == "__main__":
    main()
