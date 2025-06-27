import os
import torch
import timm
from tqdm import tqdm
from data_processing.fits_loader import load_fits_files
from data_processing.image_preprocessor import preprocess_image
from data_processing.feature_extractor import extract_features
from analysis.clustering import cluster_features
from analysis.evaluation import evaluate_kmeans
from matching.coordinate_match import find_matching_pairs
from visualization.comparison_plots import plot_and_save_matches

def main():
    # 初始化设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_folder = '/home/ubuntu/V-net/jm/cluster_results_densenet201/cluster_0'
    output_folder = '/home/ubuntu/V-net/jm/cluster_results_densenet201_0'
    radio_dir = "/home/ubuntu/V-net/jm/fits-shedian"
    
    # 1. 加载模型
    print("\n正在加载DenseNet201模型...")
    model = timm.create_model('densenet201', pretrained=False)
    model.load_state_dict(torch.load('/home/ubuntu/V-net/jm/pytorch_model_densenet201.bin'), strict=False)
    model = model.to(device)
    
    # 2. 数据处理流程
    fits_data, fits_files = load_fits_files(input_folder)
    preprocessed_images = [preprocess_image(data) for data in tqdm(fits_data, desc="图像预处理")]
    features = extract_features(model, preprocessed_images, device)
    
    # 3. 聚类分析
    evaluation_results = evaluate_kmeans(features, max_k=10)
    best_k = np.argmax(evaluation_results['silhouette']) + 2
    labels = cluster_features(features, n_clusters=best_k)
    
    # 4. 光学-射电匹配
    coord_matches = find_matching_pairs(output_folder, radio_dir)
    plot_and_save_matches(output_folder, radio_dir, coord_matches)

if __name__ == "__main__":
    main()
