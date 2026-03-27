"""
数字岩心数据集
支持连续灰度数据和离散标签数据
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import os

from am_loader import read_am_file


class DigitalRockDataset(Dataset):
    """
    数字岩心数据集
    将体素数据转换为 (坐标, 密度值) 对
    支持：
    - 连续灰度数据 (0-255)
    - 离散标签数据 (如分割后的多相数据)
    """
    def __init__(
        self,
        voxel_data: np.ndarray,
        n_samples_per_epoch: int = 1000000,
        normalize: bool = True,
        is_label_data: bool = False,
    ):
        """
        Args:
            voxel_data: (D, H, W) 体素数据
            n_samples_per_epoch: 每个epoch采样的点数
            normalize: 是否归一化密度值到 [0, 1]
            is_label_data: 是否为标签数据（自动检测：如果唯一值<20则认为是标签）
        """
        self.voxel_data = voxel_data.astype(np.float32)
        self.n_samples = n_samples_per_epoch
        self.shape = voxel_data.shape
        
        # 自动检测是否为标签数据
        unique_values = np.unique(voxel_data)
        self.is_label_data = is_label_data or len(unique_values) < 20
        self.unique_labels = unique_values
        
        if self.is_label_data:
            # 标签数据：归一化到 [0, 1] 基于最大标签值
            max_label = self.voxel_data.max()
            if max_label > 0:
                self.voxel_data = self.voxel_data / max_label
            print(f"检测到标签数据，唯一值: {unique_values}")
        elif normalize and self.voxel_data.max() > 1.0:
            # 灰度数据：归一化到 [0, 1]
            self.voxel_data = self.voxel_data / 255.0
        
        print(f"数据集形状: {self.shape}")
        print(f"归一化后值范围: [{self.voxel_data.min():.3f}, {self.voxel_data.max():.3f}]")
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 随机采样坐标
        z = np.random.randint(0, self.shape[0])
        y = np.random.randint(0, self.shape[1])
        x = np.random.randint(0, self.shape[2])
        
        # 归一化坐标到 [0, 1]
        coords = np.array([
            x / (self.shape[2] - 1),
            y / (self.shape[1] - 1),
            z / (self.shape[0] - 1),
        ], dtype=np.float32)
        
        # 获取密度值
        density = self.voxel_data[z, y, x]
        
        return torch.from_numpy(coords), torch.tensor([density], dtype=torch.float32)


class DigitalRockDatasetFull(Dataset):
    """
    全体素数据集（用于验证）
    """
    def __init__(self, voxel_data: np.ndarray, normalize: bool = True):
        self.voxel_data = voxel_data.astype(np.float32)
        self.shape = voxel_data.shape
        
        if normalize and self.voxel_data.max() > 1.0:
            self.voxel_data = self.voxel_data / 255.0
        
        # 预计算所有坐标
        D, H, W = self.shape
        z = np.linspace(0, 1, D)
        y = np.linspace(0, 1, H)
        x = np.linspace(0, 1, W)
        
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        self.coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
        self.densities = self.voxel_data.flatten()
        
    def __len__(self):
        return len(self.densities)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.coords[idx]), torch.tensor([self.densities[idx]], dtype=torch.float32)


def load_digital_rock(path: str, downsample: int = 1, crop_size: int = None) -> np.ndarray:
    """
    加载数字岩心数据
    
    Args:
        path: AM文件路径或npy文件路径
        downsample: 下采样因子
        crop_size: 裁剪尺寸（从中心裁剪立方体）
    
    Returns:
        voxel_data: (D, H, W) 体素数据
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        voxel_data = np.load(path)
    elif ext == '.am':
        voxel_data = read_am_file(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    print(f"原始数据形状: {voxel_data.shape}")
    print(f"原始数据大小: {voxel_data.nbytes / 1024 / 1024:.2f} MB")
    
    # 裁剪（用于测试或减少计算量）
    if crop_size is not None and crop_size < min(voxel_data.shape):
        D, H, W = voxel_data.shape
        d_start = (D - crop_size) // 2
        h_start = (H - crop_size) // 2
        w_start = (W - crop_size) // 2
        voxel_data = voxel_data[
            d_start:d_start+crop_size,
            h_start:h_start+crop_size,
            w_start:w_start+crop_size
        ]
        print(f"裁剪到 {crop_size}³ 后形状: {voxel_data.shape}")
    
    if downsample > 1:
        voxel_data = voxel_data[::downsample, ::downsample, ::downsample]
        print(f"下采样 {downsample}x 后形状: {voxel_data.shape}")
    
    # 打印数据统计
    unique_values = np.unique(voxel_data)
    print(f"唯一值数量: {len(unique_values)}")
    if len(unique_values) < 20:
        print(f"唯一值: {unique_values}")
    
    return voxel_data


def create_low_resolution(voxel_data: np.ndarray, factor: int) -> np.ndarray:
    """
    创建低分辨率版本（用于超分辨率实验）
    
    Args:
        voxel_data: 原始体素数据
        factor: 下采样因子
    
    Returns:
        low_res: 低分辨率体素数据
    """
    return voxel_data[::factor, ::factor, ::factor]
