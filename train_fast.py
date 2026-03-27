"""
快速训练脚本 - GPU 直接采样，无 CPU 瓶颈
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm
import time

from model import DigitalRockINR
from metrics import compute_metrics, compute_porosity, detect_dataset_type


def validate_large_dataset(model, voxel_tensor, D, H, W, device, use_gpu_data, 
                           is_label_data, max_label, dataset_type, n_samples=10_000_000):
    """
    对大数据集进行采样验证，避免内存溢出
    
    Args:
        model: INR 模型
        voxel_tensor: 体素数据 (CPU 或 GPU)
        D, H, W: 体素尺寸
        device: 计算设备
        use_gpu_data: 数据是否在 GPU
        is_label_data: 是否为标签数据
        max_label: 最大标签值
        dataset_type: 数据集类型
        n_samples: 采样点数量
    
    Returns:
        metrics: 指标字典
    """
    model.eval()
    
    # 分批采样计算
    batch_size = 1_000_000  # 每批 100 万
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_pred = []
    all_gt = []
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch = min(batch_size, n_samples - batch_idx * batch_size)
            
            # 随机采样坐标
            rand_coords = torch.rand(current_batch, 3, device=device)
            
            # 转换为体素索引
            z_idx = (rand_coords[:, 2] * (D - 1)).long()
            y_idx = (rand_coords[:, 1] * (H - 1)).long()
            x_idx = (rand_coords[:, 0] * (W - 1)).long()
            
            # 获取预测值
            pred = model(rand_coords).squeeze(-1)
            
            # 获取真实值
            if use_gpu_data:
                gt = voxel_tensor[z_idx, y_idx, x_idx]
            else:
                z_idx_cpu = z_idx.cpu()
                y_idx_cpu = y_idx.cpu()
                x_idx_cpu = x_idx.cpu()
                gt = voxel_tensor[z_idx_cpu, y_idx_cpu, x_idx_cpu].to(device)
            
            all_pred.append(pred.cpu())
            all_gt.append(gt.cpu())
    
    # 合并所有采样
    pred_samples = torch.cat(all_pred).numpy()
    gt_samples = torch.cat(all_gt).numpy()
    
    # 计算 PSNR
    mse = np.mean((pred_samples - gt_samples) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # 计算 MAE
    mae = np.mean(np.abs(pred_samples - gt_samples))
    
    # 计算 SSIM (使用切片方式)
    # 采样一些切片计算 SSIM
    from skimage.metrics import structural_similarity as ssim_func
    ssim_values = []
    for _ in range(10):
        z = np.random.randint(0, D)
        if use_gpu_data:
            gt_slice = voxel_tensor[z, :, :].cpu().numpy()
        else:
            gt_slice = voxel_tensor[z, :, :].numpy()
        
        # 重建该切片
        with torch.no_grad():
            y_coords = torch.linspace(0, 1, H, device=device)
            x_coords = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            z_val = z / (D - 1)
            coords = torch.stack([xx.flatten(), yy.flatten(), 
                                  torch.full((H*W,), z_val, device=device)], dim=-1)
            pred_slice = model(coords).reshape(H, W).cpu().numpy()
        
        s = ssim_func(gt_slice, pred_slice, data_range=1.0)
        ssim_values.append(s)
    ssim_avg = np.mean(ssim_values)
    
    # 计算标签准确率（如果是标签数据）
    metrics = {
        'psnr': float(psnr),
        'ssim': float(ssim_avg),
        'mse': float(mse),
        'mae': float(mae),
    }
    
    if is_label_data:
        pred_labels = np.round(pred_samples * max_label).astype(int)
        gt_labels = np.round(gt_samples * max_label).astype(int)
        accuracy = np.mean(pred_labels == gt_labels)
        metrics['label_accuracy'] = float(accuracy)
        
        # 各类准确率
        for label in np.unique(gt_labels):
            mask = gt_labels == label
            if mask.sum() > 0:
                metrics[f'acc_label_{label}'] = float(np.mean(pred_labels[mask] == label))
    
    # 计算孔隙度 (从采样数据估算)
    if dataset_type == 'mrccm':
        # MRCCM: 孔隙 = 宏孔(1) + 微孔(3)
        gt_labels = np.round(gt_samples * max_label).astype(int)
        pred_labels = np.round(pred_samples * max_label).astype(int)
        porosity_gt = np.mean(np.isin(gt_labels, [1, 3]))
        porosity_pred = np.mean(np.isin(pred_labels, [1, 3]))
    else:
        # 其他：假设最小值是孔隙
        porosity_gt = np.mean(gt_samples < 0.5)
        porosity_pred = np.mean(pred_samples < 0.5)
    
    metrics['porosity_gt'] = float(porosity_gt)
    metrics['porosity_pred'] = float(porosity_pred)
    metrics['porosity_error'] = float(abs(porosity_gt - porosity_pred))
    
    print(f"  Sampled {n_samples:,} points for validation")
    
    return metrics


def train_fast(
    voxel_data: np.ndarray,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 1048576,  # 100万
    lr: float = 1e-3,
    n_levels: int = 16,
    log2_hashmap_size: int = 20,
    hidden_dim: int = 64,
    validate_every: int = 20,
):
    """GPU 直接采样的快速训练，支持大数据集"""
    device = 'cuda'
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    print(f"Volume shape: {voxel_data.shape}")
    original_size = voxel_data.nbytes
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    
    # 检测标签数据和数据集类型
    unique_values = np.unique(voxel_data)
    is_label_data = len(unique_values) < 20
    max_label = float(voxel_data.max()) if voxel_data.max() > 0 else 1.0
    dataset_type = detect_dataset_type(voxel_data)
    print(f"Is label data: {is_label_data}, Max label: {max_label}")
    print(f"Unique values: {unique_values}")
    print(f"Detected dataset type: {dataset_type}")
    
    # 归一化
    voxel_data = voxel_data.astype(np.float32)
    data_min = float(voxel_data.min())
    data_max = float(voxel_data.max())
    
    if is_label_data:
        # 标签数据：除以最大标签值
        voxel_data = voxel_data / max_label
        print(f"Label data normalized by max_label={max_label}")
    elif data_max > 1.0:
        # 灰度数据：归一化到 [0, 1]
        voxel_data = (voxel_data - data_min) / (data_max - data_min)
        print(f"Grayscale data normalized: [{data_min:.1f}, {data_max:.1f}] -> [0, 1]")
    
    D, H, W = voxel_data.shape
    
    # 检查数据大小，决定是放 GPU 还是 CPU
    data_size_gb = voxel_data.nbytes / (1024**3)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # 预留 8GB 给模型和梯度
    use_gpu_data = data_size_gb < (gpu_memory_gb - 8)
    
    if use_gpu_data:
        voxel_tensor = torch.from_numpy(voxel_data).to(device)
        print(f"Data loaded to GPU, shape: {D} x {H} x {W}")
    else:
        # 大数据集：保留在 CPU
        # 对于超大数据(>20GB float32)，直接用普通内存避免 pinned memory 限制
        if data_size_gb > 20:
            voxel_tensor = torch.from_numpy(voxel_data)
            print(f"Data kept on CPU (regular memory), shape: {D} x {H} x {W}")
        else:
            # 尝试 pinned memory（更快），失败则用普通内存
            try:
                voxel_tensor = torch.from_numpy(voxel_data).pin_memory()
                print(f"Data kept on CPU (pinned memory), shape: {D} x {H} x {W}")
            except Exception as e:
                print(f"Warning: pinned memory failed, using regular memory")
                voxel_tensor = torch.from_numpy(voxel_data)
                print(f"Data kept on CPU (regular memory), shape: {D} x {H} x {W}")
        print(f"  Reason: data size ({data_size_gb:.1f} GB) + model > GPU memory ({gpu_memory_gb:.1f} GB)")
    
    # 创建模型
    model = DigitalRockINR(
        encoding_type='hash',
        n_levels=n_levels,
        log2_hashmap_size=log2_hashmap_size,
        hidden_dim=hidden_dim,
        n_hidden_layers=3,
        finest_resolution=max(D, H, W),
    ).to(device)
    
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    compression_ratio = original_size / model_size
    print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 每 epoch 的迭代次数
    total_voxels = D * H * W
    iters_per_epoch = max(total_voxels // batch_size, 10)
    print(f"Iterations per epoch: {iters_per_epoch}")
    
    best_psnr = 0.0
    train_losses = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for i in range(iters_per_epoch):
            # 在 GPU 上直接生成随机坐标
            rand_coords = torch.rand(batch_size, 3, device=device)
            
            # 转换为体素索引
            z_idx = (rand_coords[:, 2] * (D - 1)).long()
            y_idx = (rand_coords[:, 1] * (H - 1)).long()
            x_idx = (rand_coords[:, 0] * (W - 1)).long()
            
            # 获取真实值
            if use_gpu_data:
                gt_values = voxel_tensor[z_idx, y_idx, x_idx].unsqueeze(-1)
            else:
                # 数据在 CPU，索引也需要在 CPU
                z_idx_cpu = z_idx.cpu()
                y_idx_cpu = y_idx.cpu()
                x_idx_cpu = x_idx.cpu()
                gt_values = voxel_tensor[z_idx_cpu, y_idx_cpu, x_idx_cpu].to(device, non_blocking=True).unsqueeze(-1)
            
            # 前向传播
            pred = model(rand_coords)
            
            # 损失
            loss = nn.functional.mse_loss(pred, gt_values)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 每 100 次打印一次
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch} [{i+1}/{iters_per_epoch}] Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / iters_per_epoch
        train_losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 验证
        if epoch % validate_every == 0 or epoch == epochs:
            print("Validating...")
            model.eval()
            
            # 清理显存
            torch.cuda.empty_cache()
            
            # 检查数据大小决定验证策略
            data_size_gb = D * H * W * 4 / (1024**3)  # float32
            
            if data_size_gb > 10:
                # 大数据集：使用采样验证
                print(f"  Large dataset ({data_size_gb:.1f} GB), using sampled validation...")
                metrics = validate_large_dataset(
                    model, voxel_tensor, D, H, W, device, 
                    use_gpu_data, is_label_data, max_label, dataset_type,
                    n_samples=10_000_000  # 1000万采样点
                )
            else:
                # 小数据集：完整验证
                with torch.no_grad():
                    if D > 1000:
                        chunk_size = 8
                    elif D > 500:
                        chunk_size = 16
                    else:
                        chunk_size = 32
                        
                    pred_volume = model.query_volume((D, H, W), device, chunk_size=chunk_size)
                    pred_np = pred_volume.cpu().numpy() if pred_volume.is_cuda else pred_volume.numpy()
                    del pred_volume
                    torch.cuda.empty_cache()
                
                # 获取 ground truth
                if use_gpu_data:
                    gt_np = voxel_tensor.cpu().numpy()
                else:
                    gt_np = voxel_tensor.numpy()
                
                metrics = compute_metrics(pred_np, gt_np, is_label_data, max_label)
                
                # 孔隙度
                metrics['porosity_gt'] = compute_porosity(
                    gt_np, pore_labels=None, max_label=max_label, 
                    dataset_type=dataset_type
                )
                metrics['porosity_pred'] = compute_porosity(
                    pred_np, pore_labels=None, max_label=max_label,
                    dataset_type=dataset_type
                )
                metrics['porosity_error'] = abs(metrics['porosity_gt'] - metrics['porosity_pred'])
                
                del pred_np, gt_np
            
            # 打印结果
            print(f"  Porosity GT: {metrics['porosity_gt']:.4f}, Pred: {metrics['porosity_pred']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
            if 'label_accuracy' in metrics:
                print(f"  Label Accuracy: {metrics['label_accuracy']:.4f}")
            print(f"  Porosity Error: {metrics['porosity_error']:.4f}")
            
            # 保存最佳
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"  Saved best model (PSNR: {best_psnr:.2f} dB)")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # ========== 生成可视化结果 ==========
    print("\nGenerating visualizations...")
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    # 1. 损失曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"  Saved: loss_curve.png")
    
    # 2. 最终重建并保存可视化
    print("  Generating final reconstruction...")
    model.eval()
    torch.cuda.empty_cache()
    
    # 对于大数据集，只重建中间切片用于可视化
    data_size_gb = D * H * W * 4 / (1024**3)  # float32
    
    if data_size_gb > 10:
        print(f"  Large dataset ({data_size_gb:.1f} GB), generating slice visualizations only...")
        # 只重建三个正交切片
        slice_z = D // 2
        slice_y = H // 2
        slice_x = W // 2
        
        with torch.no_grad():
            # XY 切片 (z = slice_z)
            y_coords = torch.linspace(0, 1, H, device=device)
            x_coords = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            z_val = slice_z / (D - 1)
            coords_xy = torch.stack([xx.flatten(), yy.flatten(), 
                                     torch.full((H*W,), z_val, device=device)], dim=-1)
            pred_xy = model(coords_xy).reshape(H, W).cpu().numpy()
            
            # XZ 切片 (y = slice_y)
            z_coords = torch.linspace(0, 1, D, device=device)
            zz, xx = torch.meshgrid(z_coords, x_coords, indexing='ij')
            y_val = slice_y / (H - 1)
            coords_xz = torch.stack([xx.flatten(), 
                                     torch.full((D*W,), y_val, device=device),
                                     zz.flatten()], dim=-1)
            pred_xz = model(coords_xz).reshape(D, W).cpu().numpy()
            
            # YZ 切片 (x = slice_x)
            zz, yy = torch.meshgrid(z_coords, y_coords, indexing='ij')
            x_val = slice_x / (W - 1)
            coords_yz = torch.stack([torch.full((D*H,), x_val, device=device),
                                     yy.flatten(), zz.flatten()], dim=-1)
            pred_yz = model(coords_yz).reshape(D, H).cpu().numpy()
        
        # 获取 GT 切片
        if use_gpu_data:
            gt_xy = voxel_tensor[slice_z, :, :].cpu().numpy()
            gt_xz = voxel_tensor[:, slice_y, :].cpu().numpy()
            gt_yz = voxel_tensor[:, :, slice_x].cpu().numpy()
        else:
            gt_xy = voxel_tensor[slice_z, :, :].numpy()
            gt_xz = voxel_tensor[:, slice_y, :].numpy()
            gt_yz = voxel_tensor[:, :, slice_x].numpy()
        
        pred_np = None  # 不保存完整重建
        gt_np = None
    else:
        with torch.no_grad():
            if D > 1000:
                chunk_size = 8
            elif D > 500:
                chunk_size = 16
            else:
                chunk_size = 32
            pred_volume = model.query_volume((D, H, W), device, chunk_size=chunk_size)
            pred_np = pred_volume.cpu().numpy() if pred_volume.is_cuda else pred_volume.numpy()
            del pred_volume
            torch.cuda.empty_cache()
        
        # 获取 ground truth
        if use_gpu_data:
            gt_np = voxel_tensor.cpu().numpy()
        else:
            gt_np = voxel_tensor.numpy()
        
        # 保存重建体积（用于后续分析）
        np.save(os.path.join(output_dir, 'reconstruction.npy'), pred_np)
        print(f"  Saved: reconstruction.npy")
        
        # 提取切片
        slice_z = D // 2
        slice_y = H // 2
        slice_x = W // 2
        pred_xy = pred_np[slice_z, :, :]
        pred_xz = pred_np[:, slice_y, :]
        pred_yz = pred_np[:, :, slice_x]
        gt_xy = gt_np[slice_z, :, :]
        gt_xz = gt_np[:, slice_y, :]
        gt_yz = gt_np[:, :, slice_x]
    
    # 3. 重建对比图（GT vs Pred vs Error）
    diff_slice = np.abs(gt_xy - pred_xy)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    im0 = axes[0].imshow(gt_xy, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(pred_xy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Reconstruction', fontsize=12)
    axes[1].axis('off')
    
    im2 = axes[2].imshow(diff_slice, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title('Error Map', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # 误差直方图
    axes[3].hist(diff_slice.flatten(), bins=50, color='steelblue', alpha=0.8, edgecolor='black')
    axes[3].set_xlabel('Absolute Error')
    axes[3].set_ylabel('Count')
    axes[3].set_title('Error Distribution')
    axes[3].set_xlim(0, 0.5)
    
    plt.suptitle(f'Compression Results (PSNR: {best_psnr:.2f} dB, Ratio: {compression_ratio:.1f}x)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: comparison.png")
    
    # 4. 三正交切片视图
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # GT 切片
    axes[0, 0].imshow(gt_xy, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'GT - XY plane (z={slice_z})')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gt_xz, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'GT - XZ plane (y={slice_y})')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_yz, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'GT - YZ plane (x={slice_x})')
    axes[0, 2].axis('off')
    
    # Pred 切片
    axes[1, 0].imshow(pred_xy, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Pred - XY plane (z={slice_z})')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(pred_xz, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Pred - XZ plane (y={slice_y})')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(pred_yz, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Pred - YZ plane (x={slice_x})')
    axes[1, 2].axis('off')
    
    plt.suptitle('Orthogonal Slices Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slices.png'), dpi=150)
    plt.close()
    print(f"  Saved: slices.png")
    
    # 5. 如果是标签数据，生成标签对比图
    if is_label_data:
        gt_labels = np.round(gt_xy * max_label).astype(int)
        pred_labels = np.round(pred_xy * max_label).astype(int)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(gt_labels, cmap='tab10', vmin=0, vmax=int(max_label))
        axes[0].set_title('GT Labels')
        axes[0].axis('off')
        
        axes[1].imshow(pred_labels, cmap='tab10', vmin=0, vmax=int(max_label))
        axes[1].set_title('Pred Labels')
        axes[1].axis('off')
        
        # 标签差异
        label_diff = (gt_labels != pred_labels).astype(float)
        axes[2].imshow(label_diff, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Label Errors ({label_diff.mean()*100:.1f}%)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_comparison.png'), dpi=150)
        plt.close()
        print(f"  Saved: label_comparison.png")
    
    # 清理内存
    if pred_np is not None:
        del pred_np
    if gt_np is not None:
        del gt_np
    
    # 保存结果
    import json
    results = {
        'original_size_mb': float(original_size / 1024 / 1024),
        'model_size_mb': float(model_size / 1024 / 1024),
        'compression_ratio': float(compression_ratio),
        'best_psnr': float(best_psnr),
        'training_time_seconds': float(training_time),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'train_losses': [float(l) for l in train_losses],
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Visualizations: loss_curve.png, comparison.png, slices.png")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Fast INR Training (GPU sampling)')
    parser.add_argument('--data', type=str, required=True, help='Path to npy file')
    parser.add_argument('--output', type=str, default='output/fast_train', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1048576, help='Batch size (default: 1M)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_levels', type=int, default=16, help='Hash encoding levels')
    parser.add_argument('--log2_hashmap_size', type=int, default=20, help='Log2 hashmap size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--validate_every', type=int, default=20, help='Validate every N epochs')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading: {args.data}")
    voxel_data = np.load(args.data)
    
    train_fast(
        voxel_data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        hidden_dim=args.hidden_dim,
        validate_every=args.validate_every,
    )


if __name__ == '__main__':
    main()
