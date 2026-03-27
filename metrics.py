"""
评估指标
支持连续灰度数据和离散标签数据
"""

import numpy as np
from typing import Dict, Optional, List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """计算 PSNR"""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """计算 SSIM (3D)"""
    # 对每个切片计算 SSIM，然后平均
    ssim_values = []
    for i in range(pred.shape[0]):
        s = ssim(pred[i], gt[i], data_range=data_range)
        ssim_values.append(s)
    return np.mean(ssim_values)


def compute_porosity(volume: np.ndarray, threshold: float = 0.5, 
                     pore_labels: Optional[List[int]] = None,
                     is_normalized: bool = True,
                     max_label: float = 7.0,
                     dataset_type: str = 'auto') -> float:
    """
    计算孔隙度
    
    Args:
        volume: 体素数据
        threshold: 孔隙阈值（用于连续数据）
        pore_labels: 孔隙标签列表（用于标签数据，如 [0] 表示标签0是孔隙）
        is_normalized: 数据是否已归一化
        max_label: 最大标签值（用于反归一化）
        dataset_type: 数据集类型 ('mrccm', 'berea', 'auto')
                     - 'mrccm': 孔隙标签为 1 (宏孔) 和 3 (微孔)
                     - 'berea': 孔隙标签为 0
                     - 'auto': 自动检测
    
    Returns:
        porosity: 孔隙度
    """
    # 根据数据集类型确定孔隙标签
    if pore_labels is None:
        if dataset_type == 'mrccm':
            # MRCCM: 1=macropores, 2=solid, 3=microporosity
            pore_labels = [1, 3]  # 宏孔 + 微孔
        elif dataset_type == 'berea':
            pore_labels = [0]  # Berea 假设 0 是孔隙
        elif dataset_type == 'auto':
            # 自动检测：检查标签值
            if is_normalized:
                test_labels = np.unique(np.round(volume * max_label).astype(int))
            else:
                test_labels = np.unique(volume.astype(int))
            
            # MRCCM 数据标签是 [1, 2, 3]
            if set(test_labels) == {1, 2, 3} or set(test_labels) <= {1, 2, 3}:
                pore_labels = [1, 3]  # MRCCM 格式
            # Berea 数据标签包含 0
            elif 0 in test_labels:
                pore_labels = [0]  # Berea 格式
            else:
                # 默认：最小标签为孔隙
                pore_labels = [int(test_labels.min())]
    
    if pore_labels is not None:
        # 标签数据：反归一化后计算
        if is_normalized:
            volume_labels = np.round(volume * max_label).astype(int)
        else:
            volume_labels = volume.astype(int)
        
        pore_voxels = sum((volume_labels == label).sum() for label in pore_labels)
    else:
        # 连续数据：使用阈值
        pore_voxels = (volume < threshold).sum()
    
    total_voxels = volume.size
    porosity = pore_voxels / total_voxels
    
    return porosity


def detect_dataset_type(volume: np.ndarray) -> str:
    """
    自动检测数据集类型
    
    Args:
        volume: 原始（未归一化）体素数据
    
    Returns:
        dataset_type: 'mrccm', 'berea', 或 'grayscale'
    """
    unique_vals = np.unique(volume)
    
    # MRCCM: 标签 [1, 2, 3]
    if set(unique_vals) == {1, 2, 3} or set(unique_vals) <= {1, 2, 3}:
        return 'mrccm'
    
    # Berea: 标签包含 0, 2, 3, 4, 5, 6, 7
    if 0 in unique_vals and len(unique_vals) <= 10:
        return 'berea'
    
    # 灰度数据
    if len(unique_vals) > 10:
        return 'grayscale'
    
    return 'auto'


def compute_label_accuracy(pred: np.ndarray, gt: np.ndarray, 
                           max_label: float = 7.0) -> Dict[str, float]:
    """
    计算标签分类准确率（用于分割数据）
    
    Args:
        pred: 预测体积（归一化）
        gt: 真实体积（归一化）
        max_label: 最大标签值
    
    Returns:
        准确率指标
    """
    # 反归一化并四舍五入到最近的标签
    pred_labels = np.round(pred * max_label).astype(int)
    gt_labels = np.round(gt * max_label).astype(int)
    
    # 总体准确率
    accuracy = (pred_labels == gt_labels).mean()
    
    # 每类准确率
    unique_labels = np.unique(gt_labels)
    per_class_acc = {}
    for label in unique_labels:
        mask = gt_labels == label
        if mask.sum() > 0:
            per_class_acc[f'acc_label_{label}'] = (pred_labels[mask] == label).mean()
    
    return {
        'label_accuracy': accuracy,
        **per_class_acc
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray, 
                    is_label_data: bool = False,
                    max_label: float = 7.0) -> Dict[str, float]:
    """
    计算所有指标
    
    Args:
        pred: 预测体积
        gt: 真实体积
        is_label_data: 是否为标签数据
        max_label: 最大标签值
    
    Returns:
        metrics: 指标字典
    """
    # 确保数据范围一致
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    
    metrics = {
        'psnr': compute_psnr(pred, gt),
        'ssim': compute_ssim(pred, gt),
        'mse': float(np.mean((pred - gt) ** 2)),
        'mae': float(np.mean(np.abs(pred - gt))),
    }
    
    # 对于标签数据，额外计算分类准确率
    if is_label_data:
        label_metrics = compute_label_accuracy(pred, gt, max_label)
        metrics.update(label_metrics)
    
    return metrics


def compute_pore_size_distribution(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    计算孔隙尺寸分布（简化版本）
    
    Args:
        volume: 体素数据
        threshold: 孔隙阈值
    
    Returns:
        psd: 孔隙尺寸分布
    """
    try:
        from scipy import ndimage
        
        # 二值化
        binary = (volume < threshold).astype(np.uint8)
        
        # 连通区域标记
        labeled, num_features = ndimage.label(binary)
        
        # 计算每个孔隙的大小
        pore_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        
        return np.array(pore_sizes)
    except ImportError:
        print("警告: scipy 未安装，无法计算孔隙尺寸分布")
        return np.array([])


def compare_pore_size_distribution(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    比较孔隙尺寸分布
    """
    pred_psd = compute_pore_size_distribution(pred, threshold)
    gt_psd = compute_pore_size_distribution(gt, threshold)
    
    if len(pred_psd) == 0 or len(gt_psd) == 0:
        return {'psd_error': float('nan')}
    
    # 计算分布差异（使用直方图）
    max_size = max(pred_psd.max(), gt_psd.max())
    bins = np.linspace(0, max_size, 50)
    
    pred_hist, _ = np.histogram(pred_psd, bins=bins, density=True)
    gt_hist, _ = np.histogram(gt_psd, bins=bins, density=True)
    
    # Wasserstein 距离（L1）
    psd_error = np.sum(np.abs(pred_hist - gt_hist)) * (bins[1] - bins[0])
    
    return {
        'psd_error': psd_error,
        'num_pores_pred': len(pred_psd),
        'num_pores_gt': len(gt_psd),
    }
