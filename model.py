"""
Implicit Neural Representation for Digital Rock
基于 InstantNGP 风格的 Hash Encoding + MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalEncoding(nn.Module):
    """位置编码（备用，如果没有 tiny-cuda-nn）"""
    def __init__(self, n_frequencies: int = 10):
        super().__init__()
        self.n_frequencies = n_frequencies
        # 频率从 2^0 到 2^(n-1)
        freqs = 2.0 ** torch.arange(n_frequencies)
        self.register_buffer('freqs', freqs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) 坐标
        Returns:
            encoded: (N, 3 * 2 * n_frequencies)
        """
        # x: (N, 3) -> (N, 3, 1) -> (N, 3, n_freq)
        x = x.unsqueeze(-1) * self.freqs * np.pi
        # sin 和 cos
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        # (N, 3, 2*n_freq) -> (N, 3*2*n_freq)
        return encoded.view(x.shape[0], -1)
    
    @property
    def output_dim(self):
        return 3 * 2 * self.n_frequencies


class HashEncoding(nn.Module):
    """
    多分辨率哈希编码 (InstantNGP 风格)
    简化版实现，使用 PyTorch
    """
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # 计算每层分辨率
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(np.ceil(base_resolution * (b ** i))) for i in range(n_levels)]
        
        # 哈希表（可学习参数）
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features_per_level) * 0.01)
            for _ in range(n_levels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) 归一化坐标 [0, 1]
        Returns:
            features: (N, n_levels * n_features_per_level)
        """
        # 确保坐标在 [0, 1] 范围
        x = torch.clamp(x, 0.0, 1.0 - 1e-6)
        
        all_features = []
        
        for level, resolution in enumerate(self.resolutions):
            # 缩放到当前分辨率
            scaled = x * resolution
            
            # 获取体素角点索引
            floor_coords = scaled.floor().long()
            
            # 三线性插值权重
            weights = scaled - floor_coords.float()
            
            # 8个角点的特征
            features = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # 角点坐标
                        corner = floor_coords + torch.tensor([i, j, k], device=x.device)
                        
                        # 哈希索引
                        hash_idx = self._hash(corner, level)
                        
                        # 插值权重
                        w = (weights[:, 0] if i else (1 - weights[:, 0])) * \
                            (weights[:, 1] if j else (1 - weights[:, 1])) * \
                            (weights[:, 2] if k else (1 - weights[:, 2]))
                        
                        # 累加特征
                        features += w.unsqueeze(-1) * self.hash_tables[level][hash_idx]
            
            all_features.append(features)
        
        return torch.cat(all_features, dim=-1)
    
    def _hash(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """空间哈希函数"""
        # 使用素数进行哈希
        primes = [1, 2654435761, 805459861]
        result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(3):
            result ^= coords[:, i] * primes[i]
        return result % self.hashmap_size
    
    @property
    def output_dim(self):
        return self.n_levels * self.n_features_per_level


class DigitalRockINR(nn.Module):
    """
    数字岩心隐式神经表示
    输入: 3D坐标 (x, y, z) ∈ [0, 1]^3
    输出: 密度/标签值
    """
    def __init__(
        self,
        encoding_type: str = 'hash',  # 'hash' or 'sinusoidal'
        n_hidden_layers: int = 3,
        hidden_dim: int = 64,
        # Hash encoding 参数
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        # Sinusoidal encoding 参数
        n_frequencies: int = 10,
    ):
        super().__init__()
        
        self.encoding_type = encoding_type
        
        # 编码器
        if encoding_type == 'hash':
            self.encoder = HashEncoding(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
            )
        else:
            self.encoder = SinusoidalEncoding(n_frequencies=n_frequencies)
        
        # MLP
        layers = []
        in_dim = self.encoder.output_dim
        
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) 归一化坐标
        Returns:
            density: (N, 1) 密度值
        """
        features = self.encoder(coords)
        return self.mlp(features)
    
    def query_volume(self, resolution: Tuple[int, int, int], device: str = 'cuda', 
                     chunk_size: int = 64) -> torch.Tensor:
        """
        查询整个体积（分块处理，避免显存溢出）
        
        Args:
            resolution: (D, H, W) 输出分辨率
            device: 设备
            chunk_size: 每次处理的深度层数
        Returns:
            volume: (D, H, W) 体素数据
        """
        D, H, W = resolution
        
        # 在 CPU 上创建输出体积
        volume = torch.zeros(D, H, W, dtype=torch.float32)
        
        # 预生成 x, y 坐标（这两个维度较小）
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xy_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)
        
        # 分块处理 z 方向
        with torch.no_grad():
            for z_start in range(0, D, chunk_size):
                z_end = min(z_start + chunk_size, D)
                chunk_d = z_end - z_start
                
                # 生成该块的 z 坐标
                z_vals = torch.linspace(z_start / D, (z_end - 1) / D, chunk_d, device=device)
                
                # 为每个 z 层创建完整坐标
                chunk_densities = []
                for z_idx, z_val in enumerate(z_vals):
                    # 创建当前层的坐标 (H*W, 3)
                    z_coord = torch.full((H * W, 1), z_val.item(), device=device)
                    coords = torch.cat([xy_coords, z_coord], dim=-1)  # (H*W, 3) - [x, y, z]
                    
                    # 分批查询
                    batch_size = 256 * 256
                    layer_density = []
                    for i in range(0, coords.shape[0], batch_size):
                        batch_coords = coords[i:i+batch_size]
                        batch_density = self(batch_coords)
                        layer_density.append(batch_density.cpu())
                    
                    layer_density = torch.cat(layer_density, dim=0).reshape(H, W)
                    chunk_densities.append(layer_density)
                
                # 组装该块
                volume[z_start:z_end] = torch.stack(chunk_densities, dim=0)
                
                # 打印进度
                if (z_start // chunk_size) % 5 == 0:
                    print(f"  Reconstructing: {z_end}/{D} slices ({100*z_end/D:.1f}%)")
        
        # 对于大体积，保留在 CPU 上避免 OOM
        volume_size_gb = volume.numel() * volume.element_size() / (1024**3)
        if volume_size_gb > 20:  # 超过 20 GB 的体积保留在 CPU
            return volume  # 返回 CPU tensor
        else:
            return volume.to(device)


class DigitalRockINRTinyCuda(nn.Module):
    """
    使用 tiny-cuda-nn 的高效版本（可选）
    需要安装: pip install tiny-cuda-nn
    """
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        try:
            import tinycudann as tcnn
            self.use_tcnn = True
            
            # Hash encoding 配置
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1)),
            }
            
            # MLP 配置
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            }
            
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config=encoding_config,
                network_config=network_config,
            )
            
        except ImportError:
            print("警告: tiny-cuda-nn 未安装，使用纯 PyTorch 实现")
            self.use_tcnn = False
            self.model = DigitalRockINR(
                encoding_type='hash',
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                n_hidden_layers=n_hidden_layers,
                hidden_dim=hidden_dim,
            )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_tcnn:
            return self.model(coords)
        else:
            return self.model(coords)
    
    def query_volume(self, resolution: Tuple[int, int, int], device: str = 'cuda') -> torch.Tensor:
        D, H, W = resolution
        
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        batch_size = 1024 * 1024
        densities = []
        
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i:i+batch_size]
                if self.use_tcnn:
                    batch_density = self.model(batch_coords.float())
                else:
                    batch_density = self.model(batch_coords)
                densities.append(batch_density)
        
        density = torch.cat(densities, dim=0)
        return density.reshape(D, H, W)
