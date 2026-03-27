"""
超分辨率实验脚本
训练低分辨率数据，推理高分辨率，与 Ground Truth 对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import json

from model import DigitalRockINR
from metrics import compute_metrics, compute_porosity, detect_dataset_type


def normalize_data(data):
    """归一化数据到 [0, 1]"""
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    return data, data_min, data_max


def train_sr(
    lr_data_path: str,
    hr_data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 2**20,
    lr: float = 1e-3,
    n_levels: int = 16,
    log2_hashmap_size: int = 19,
    hidden_dim: int = 64,
    validate_every: int = 25,
    device: str = 'cuda'
):
    """
    超分辨率训练
    
    Args:
        lr_data_path: 低分辨率数据路径
        hr_data_path: 高分辨率 ground truth 路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    lr_data = np.load(lr_data_path)
    hr_data = np.load(hr_data_path)
    
    print(f"LR data: {lr_data.shape}, dtype: {lr_data.dtype}")
    print(f"HR data: {hr_data.shape}, dtype: {hr_data.dtype}")
    
    # 计算放大倍数
    scale_factors = [hr_data.shape[i] / lr_data.shape[i] for i in range(3)]
    print(f"Scale factors: {scale_factors}")
    
    # 检查是否是分割数据
    lr_unique = np.unique(lr_data)
    hr_unique = np.unique(hr_data)
    print(f"LR unique values: {lr_unique[:10]}{'...' if len(lr_unique) > 10 else ''}")
    print(f"HR unique values: {hr_unique}")
    
    lr_is_segmented = len(lr_unique) <= 10
    hr_is_segmented = len(hr_unique) <= 10
    if lr_is_segmented:
        print(f"LR data is segmented with {len(lr_unique)} labels")
    if hr_is_segmented:
        print(f"HR data is segmented with {len(hr_unique)} labels")
    
    # 归一化 LR 数据
    lr_normalized, lr_min, lr_max = normalize_data(lr_data)
    print(f"LR normalized range: [{lr_normalized.min():.3f}, {lr_normalized.max():.3f}]")
    
    # 转为 tensor 并移到 GPU
    D_lr, H_lr, W_lr = lr_data.shape
    D_hr, H_hr, W_hr = hr_data.shape
    
    lr_tensor = torch.from_numpy(lr_normalized).float().to(device)
    
    # 计算大小
    lr_size = lr_data.nbytes
    hr_size = hr_data.nbytes
    print(f"LR size: {lr_size / 1024 / 1024:.2f} MB")
    print(f"HR size: {hr_size / 1024 / 1024:.2f} MB")
    
    # 创建模型
    model = DigitalRockINR(
        n_levels=n_levels,
        log2_hashmap_size=log2_hashmap_size,
        hidden_dim=hidden_dim
    ).to(device)
    
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # 预计算总采样数
    total_voxels = D_lr * H_lr * W_lr
    samples_per_epoch = min(total_voxels, batch_size * 100)
    batches_per_epoch = samples_per_epoch // batch_size
    
    print(f"\nTraining config:")
    print(f"  Total LR voxels: {total_voxels:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Epochs: {epochs}")
    
    # 训练
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    start_time = time.time()
    train_losses = []
    best_psnr = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch_idx in range(batches_per_epoch):
            # 随机采样坐标
            indices = torch.randint(0, total_voxels, (batch_size,), device=device)
            
            # 转换为 3D 坐标
            z_idx = indices // (H_lr * W_lr)
            y_idx = (indices % (H_lr * W_lr)) // W_lr
            x_idx = indices % W_lr
            
            # 归一化坐标到 [0, 1]
            coords = torch.stack([
                x_idx.float() / (W_lr - 1),
                y_idx.float() / (H_lr - 1),
                z_idx.float() / (D_lr - 1)
            ], dim=-1)
            
            # 获取目标值
            targets = lr_tensor[z_idx, y_idx, x_idx].unsqueeze(-1)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(coords)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        scheduler.step()
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 验证（在 HR 分辨率）
        if epoch % validate_every == 0 or epoch == epochs:
            print(f"\nValidating at epoch {epoch}...")
            model.eval()
            torch.cuda.empty_cache()
            
            # 检测 HR 数据大小，决定使用全量还是采样验证
            hr_size_gb = hr_data.nbytes / (1024**3)
            use_sampled_validation = hr_size_gb > 5.0  # 超过 5GB 使用采样验证
            
            if use_sampled_validation:
                print(f"  HR data {hr_size_gb:.1f} GB > 5GB, using sampled validation...")
                n_samples = 50_000_000  # 5000 万个采样点
                
                with torch.no_grad():
                    # 随机采样 HR 坐标
                    z_idx = torch.randint(0, D_hr, (n_samples,))
                    y_idx = torch.randint(0, H_hr, (n_samples,))
                    x_idx = torch.randint(0, W_hr, (n_samples,))
                    
                    # 获取 HR ground truth
                    hr_samples = hr_data[z_idx.numpy(), y_idx.numpy(), x_idx.numpy()]
                    
                    # 归一化坐标
                    z_norm = z_idx.float() / (D_hr - 1)
                    y_norm = y_idx.float() / (H_hr - 1)
                    x_norm = x_idx.float() / (W_hr - 1)
                    
                    # 分批推理（注意：模型坐标顺序为 [x, y, z]）
                    batch_infer = 2_000_000
                    sr_samples = []
                    for i in range(0, n_samples, batch_infer):
                        end_idx = min(i + batch_infer, n_samples)
                        coords_batch = torch.stack([
                            x_norm[i:end_idx], y_norm[i:end_idx], z_norm[i:end_idx]
                        ], dim=-1).to(device)
                        pred = model(coords_batch).squeeze(-1).cpu()
                        sr_samples.append(pred)
                        del coords_batch
                    
                    sr_samples = torch.cat(sr_samples).numpy()
                    torch.cuda.empty_cache()
                
                # 反归一化：sr_samples 从 [0,1] 恢复到 [lr_min, lr_max] 即 [1, 3]
                sr_samples = sr_samples * (lr_max - lr_min) + lr_min
                
                if hr_is_segmented:
                    # 直接四舍五入得到预测标签（sr_samples 已经是 [1,3] 范围）
                    sr_pred_labels = np.round(sr_samples).astype(np.uint8)
                    sr_pred_labels = np.clip(sr_pred_labels, 1, int(lr_max))
                    
                    accuracy = np.mean(sr_pred_labels == hr_samples)
                    print(f"  Segmentation Accuracy (sampled): {accuracy:.4f}")
                    
                    for label in hr_unique:
                        mask = hr_samples == label
                        if mask.sum() > 0:
                            label_acc = np.mean(sr_pred_labels[mask] == label)
                            print(f"    Label {label} Accuracy: {label_acc:.4f}")
                    
                    metrics = {'segmentation_accuracy': float(accuracy)}
                    
                    hr_dataset_type = detect_dataset_type(hr_data)
                    print(f"  HR dataset type: {hr_dataset_type}")
                    
                    if hr_dataset_type == 'mrccm':
                        pore_labels = [1, 3]
                        porosity_gt = np.mean(np.isin(hr_samples, pore_labels))
                        porosity_pred = np.mean(np.isin(sr_pred_labels, pore_labels))
                    else:
                        pore_label = hr_unique.min()
                        porosity_gt = np.mean(hr_samples == pore_label)
                        porosity_pred = np.mean(sr_pred_labels == pore_label)
                    
                    metrics['porosity_gt'] = float(porosity_gt)
                    metrics['porosity_pred'] = float(porosity_pred)
                    metrics['porosity_error'] = float(abs(porosity_gt - porosity_pred))
                    print(f"  Porosity - GT: {porosity_gt:.4f}, Pred: {porosity_pred:.4f}, Error: {metrics['porosity_error']:.4f}")
                    current_metric = accuracy
                    
                    del sr_pred_labels
                else:
                    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                    psnr = peak_signal_noise_ratio(hr_samples.astype(np.float32), sr_samples.astype(np.float32), data_range=hr_samples.max() - hr_samples.min())
                    metrics = {'psnr': psnr}
                    print(f"  PSNR (sampled): {psnr:.2f} dB")
                    current_metric = psnr
                
                del sr_samples, hr_samples
                
            else:
                # 小数据集使用全量验证
                with torch.no_grad():
                    print(f"  Reconstructing at HR resolution {D_hr}x{H_hr}x{W_hr}...")
                    chunk_size = 16
                    sr_volume = model.query_volume((D_hr, H_hr, W_hr), device, chunk_size=chunk_size)
                    sr_np = sr_volume.cpu().numpy()
                    del sr_volume
                    torch.cuda.empty_cache()
                
                sr_np = sr_np * (lr_max - lr_min) + lr_min
                
                if hr_is_segmented:
                    sr_segmented = segment_volume(sr_np, n_classes=len(hr_unique), max_label=lr_max, lr_is_segmented=lr_is_segmented)
                    
                    accuracy = np.mean(sr_segmented == hr_data)
                    print(f"  Segmentation Accuracy: {accuracy:.4f}")
                    
                    for label in hr_unique:
                        mask = hr_data == label
                        if mask.sum() > 0:
                            label_acc = np.mean(sr_segmented[mask] == label)
                            print(f"    Label {label} Accuracy: {label_acc:.4f}")
                    
                    metrics = {'segmentation_accuracy': float(accuracy)}
                    
                    hr_dataset_type = detect_dataset_type(hr_data)
                    print(f"  HR dataset type: {hr_dataset_type}")
                    
                    if hr_dataset_type == 'mrccm':
                        pore_labels = [1, 3]
                        porosity_gt = np.mean(np.isin(hr_data, pore_labels))
                        porosity_pred = np.mean(np.isin(sr_segmented, pore_labels))
                    else:
                        pore_label = hr_unique.min()
                        porosity_gt = np.mean(hr_data == pore_label)
                        porosity_pred = np.mean(sr_segmented == pore_label)
                    
                    metrics['porosity_gt'] = float(porosity_gt)
                    metrics['porosity_pred'] = float(porosity_pred)
                    metrics['porosity_error'] = float(abs(porosity_gt - porosity_pred))
                    print(f"  Porosity - GT: {porosity_gt:.4f}, Pred: {porosity_pred:.4f}, Error: {metrics['porosity_error']:.4f}")
                    current_metric = accuracy
                    del sr_segmented
                else:
                    metrics = compute_metrics(sr_np, hr_data.astype(np.float32))
                    print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                    current_metric = metrics['psnr']
                
                del sr_np
            
            # 保存最佳模型
            if current_metric > best_psnr:
                best_psnr = current_metric
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"  Saved best model (metric: {best_psnr:.4f})")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # ========== 生成可视化结果 ==========
    print("\nGenerating visualizations...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom as scipy_zoom
    
    # 1. 损失曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Super-Resolution Training Loss', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"  Saved: loss_curve.png")
    
    # 2. 最终 SR 结果
    print("  Generating final SR reconstruction...")
    model.eval()
    torch.cuda.empty_cache()
    hr_D, hr_H, hr_W = hr_data.shape
    
    # 检查 HR 数据大小，决定生成方式
    hr_size_gb = hr_data.nbytes / (1024**3)
    skip_full_reconstruction = hr_size_gb > 5.0
    
    if skip_full_reconstruction:
        print(f"  HR data {hr_size_gb:.1f} GB > 5GB, skipping full reconstruction...")
        print(f"  Only generating orthogonal slices for visualization...")
        sr_np = None
        
        # 只生成三个正交切片
        slice_z = hr_D // 2
        slice_y = hr_H // 2
        slice_x = hr_W // 2
        
        with torch.no_grad():
            # XY 切片 (固定 z)，坐标顺序 [x, y, z]
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(0, 1, hr_H),
                torch.linspace(0, 1, hr_W),
                indexing='ij'
            )
            z_val = slice_z / (hr_D - 1)
            coords_xy = torch.stack([
                x_coords, y_coords, torch.full_like(y_coords, z_val)
            ], dim=-1).reshape(-1, 3).to(device)
            
            sr_xy = []
            for i in range(0, coords_xy.shape[0], 1000000):
                pred = model(coords_xy[i:i+1000000]).squeeze(-1).cpu()
                sr_xy.append(pred)
            sr_xy = torch.cat(sr_xy).reshape(hr_H, hr_W).numpy()
            del coords_xy
            torch.cuda.empty_cache()
            
            # XZ 切片 (固定 y)，坐标顺序 [x, y, z]
            z_coords, x_coords = torch.meshgrid(
                torch.linspace(0, 1, hr_D),
                torch.linspace(0, 1, hr_W),
                indexing='ij'
            )
            y_val = slice_y / (hr_H - 1)
            coords_xz = torch.stack([
                x_coords,
                torch.full_like(z_coords, y_val),
                z_coords
            ], dim=-1).reshape(-1, 3).to(device)
            
            sr_xz = []
            for i in range(0, coords_xz.shape[0], 1000000):
                pred = model(coords_xz[i:i+1000000]).squeeze(-1).cpu()
                sr_xz.append(pred)
            sr_xz = torch.cat(sr_xz).reshape(hr_D, hr_W).numpy()
            del coords_xz
            torch.cuda.empty_cache()
            
            # YZ 切片 (固定 x)，坐标顺序 [x, y, z]
            z_coords, y_coords = torch.meshgrid(
                torch.linspace(0, 1, hr_D),
                torch.linspace(0, 1, hr_H),
                indexing='ij'
            )
            x_val = slice_x / (hr_W - 1)
            coords_yz = torch.stack([
                torch.full_like(z_coords, x_val),
                y_coords, z_coords
            ], dim=-1).reshape(-1, 3).to(device)
            
            sr_yz = []
            for i in range(0, coords_yz.shape[0], 1000000):
                pred = model(coords_yz[i:i+1000000]).squeeze(-1).cpu()
                sr_yz.append(pred)
            sr_yz = torch.cat(sr_yz).reshape(hr_D, hr_H).numpy()
            del coords_yz
            torch.cuda.empty_cache()
        
        # 反归一化切片
        sr_xy = sr_xy * (lr_max - lr_min) + lr_min
        sr_xz = sr_xz * (lr_max - lr_min) + lr_min
        sr_yz = sr_yz * (lr_max - lr_min) + lr_min
        
        # 获取 GT 切片
        gt_xy = hr_data[slice_z].astype(np.float32)
        gt_xz = hr_data[:, slice_y, :].astype(np.float32)
        gt_yz = hr_data[:, :, slice_x].astype(np.float32)
        
        # 保存切片
        np.savez(os.path.join(output_dir, 'sr_slices.npz'),
                 sr_xy=sr_xy, sr_xz=sr_xz, sr_yz=sr_yz,
                 gt_xy=gt_xy, gt_xz=gt_xz, gt_yz=gt_yz)
        print(f"  Saved: sr_slices.npz")
        
        # 生成 bicubic 切片用于对比
        lr_normalized_slice_z = lr_data[lr_data.shape[0]//2].astype(np.float32)
        bicubic_xy = scipy_zoom(lr_normalized_slice_z, [scale_factors[1], scale_factors[2]], order=3)
        bicubic_xy = np.clip(bicubic_xy, 0, lr_data.max())
        # 确保形状与 HR 切片一致（scipy_zoom 可能有 ±1 的误差）
        bicubic_xy = bicubic_xy[:hr_H, :hr_W]
        
    else:
        with torch.no_grad():
            chunk_size = 32 if hr_D > 500 else 64
            sr_volume = model.query_volume((hr_D, hr_H, hr_W), device, chunk_size=chunk_size)
            sr_np = sr_volume.cpu().numpy()
            del sr_volume
            torch.cuda.empty_cache()
        
        # 保存 SR 结果
        np.save(os.path.join(output_dir, 'sr_result.npy'), sr_np)
        print(f"  Saved: sr_result.npy")
    
    # 只对小数据集生成 Bicubic 基线
    if not skip_full_reconstruction:
        bicubic_np = scipy_zoom(lr_data.astype(np.float32), scale_factors, order=3)
        bicubic_np = np.clip(bicubic_np, 0, 1)
    
    # 3. 超分辨率对比图（HR GT vs Bicubic vs INR）
    if skip_full_reconstruction:
        # 大数据集：只用切片生成对比图
        hr_max_val = hr_data.max()
        gt_xy_norm = gt_xy / hr_max_val if hr_is_segmented else gt_xy
        sr_xy_norm = sr_xy / hr_max_val if hr_is_segmented else sr_xy
        bicubic_xy_norm = bicubic_xy / hr_max_val if hr_is_segmented else bicubic_xy
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].imshow(gt_xy_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('HR Ground Truth')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bicubic_xy_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Bicubic Interpolation')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sr_xy_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('INR Super-Resolution')
        axes[0, 2].axis('off')
        
        inr_error = np.abs(gt_xy_norm - sr_xy_norm)
        bicubic_error = np.abs(gt_xy_norm - bicubic_xy_norm)
        im = axes[0, 3].imshow(inr_error, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 3].set_title('INR Error Map')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
        
        h, w = gt_xy_norm.shape
        crop_size = min(h, w) // 4
        y1, x1 = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        y2, x2 = y1 + crop_size, x1 + crop_size
        
        axes[1, 0].imshow(gt_xy_norm[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('HR (Zoomed)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(bicubic_xy_norm[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Bicubic (Zoomed)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(sr_xy_norm[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('INR (Zoomed)')
        axes[1, 2].axis('off')
        
        axes[1, 3].hist(bicubic_error.flatten(), bins=50, alpha=0.6, label='Bicubic', color='gray')
        axes[1, 3].hist(inr_error.flatten(), bins=50, alpha=0.6, label='INR', color='red')
        axes[1, 3].set_xlabel('Absolute Error')
        axes[1, 3].set_ylabel('Count')
        axes[1, 3].set_title('Error Distribution')
        axes[1, 3].legend()
        axes[1, 3].set_xlim(0, 0.5)
        
        plt.suptitle(f'Super-Resolution Comparison ({scale_factors[0]:.0f}x upscaling) - XY Slice', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sr_comparison.png'), dpi=150)
        plt.close()
        print(f"  Saved: sr_comparison.png")
        
        # 标签对比图
        if hr_is_segmented:
            sr_labels_xy = np.round(sr_xy).astype(np.uint8)
            sr_labels_xy = np.clip(sr_labels_xy, 1, int(hr_max_val))
            bicubic_labels_xy = np.round(bicubic_xy).astype(np.uint8)
            bicubic_labels_xy = np.clip(bicubic_labels_xy, 1, int(hr_max_val))
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            axes[0, 0].imshow(gt_xy.astype(np.uint8), cmap='tab10', vmin=0, vmax=3)
            axes[0, 0].set_title('HR GT Labels')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(bicubic_labels_xy, cmap='tab10', vmin=0, vmax=3)
            axes[0, 1].set_title('Bicubic Labels')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(sr_labels_xy, cmap='tab10', vmin=0, vmax=3)
            axes[0, 2].set_title('INR Labels')
            axes[0, 2].axis('off')
            
            bicubic_label_error = (bicubic_labels_xy != gt_xy.astype(np.uint8)).astype(float)
            inr_label_error = (sr_labels_xy != gt_xy.astype(np.uint8)).astype(float)
            
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(bicubic_label_error, cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Bicubic Error ({bicubic_label_error.mean()*100:.1f}%)')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(inr_label_error, cmap='Reds', vmin=0, vmax=1)
            axes[1, 2].set_title(f'INR Error ({inr_label_error.mean()*100:.1f}%)')
            axes[1, 2].axis('off')
            
            plt.suptitle('Label Comparison (Segmented Data) - XY Slice', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'label_comparison.png'), dpi=150)
            plt.close()
            print(f"  Saved: label_comparison.png")
        
        # 计算基于切片的 bicubic metrics
        bicubic_metrics = {
            'psnr': float(peak_signal_noise_ratio(gt_xy_norm, bicubic_xy_norm.astype(np.float32), data_range=1.0) if 'peak_signal_noise_ratio' in dir() else 0),
        }
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        bicubic_metrics['psnr'] = float(peak_signal_noise_ratio(gt_xy_norm.astype(np.float32), bicubic_xy_norm.astype(np.float32), data_range=1.0))
        bicubic_metrics['ssim'] = float(structural_similarity(gt_xy_norm.astype(np.float32), bicubic_xy_norm.astype(np.float32), data_range=1.0))
        print(f"\nBicubic baseline (slice) - PSNR: {bicubic_metrics['psnr']:.2f} dB, SSIM: {bicubic_metrics['ssim']:.4f}")
        
    else:
        # 小数据集：完整重建
        hr_norm = hr_data.astype(np.float32)
        if hr_is_segmented:
            hr_norm = hr_norm / hr_data.max()
        
        slice_idx = hr_D // 2
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 上排：完整切片
        axes[0, 0].imshow(hr_norm[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('HR Ground Truth')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bicubic_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Bicubic Interpolation')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sr_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('INR Super-Resolution')
        axes[0, 2].axis('off')
        
        # 误差图
        inr_error = np.abs(hr_norm[slice_idx] - sr_np[slice_idx])
        bicubic_error = np.abs(hr_norm[slice_idx] - bicubic_np[slice_idx])
        im = axes[0, 3].imshow(inr_error, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 3].set_title('INR Error Map')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
        
        # 下排：放大区域
        h, w = hr_norm[slice_idx].shape
        crop_size = min(h, w) // 4
        y1, x1 = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        y2, x2 = y1 + crop_size, x1 + crop_size
        
        axes[1, 0].imshow(hr_norm[slice_idx, y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('HR (Zoomed)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(bicubic_np[slice_idx, y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Bicubic (Zoomed)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(sr_np[slice_idx, y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('INR (Zoomed)')
        axes[1, 2].axis('off')
        
        # 误差直方图对比
        axes[1, 3].hist(bicubic_error.flatten(), bins=50, alpha=0.6, label='Bicubic', color='gray')
        axes[1, 3].hist(inr_error.flatten(), bins=50, alpha=0.6, label='INR', color='red')
        axes[1, 3].set_xlabel('Absolute Error')
        axes[1, 3].set_ylabel('Count')
        axes[1, 3].set_title('Error Distribution')
        axes[1, 3].legend()
        axes[1, 3].set_xlim(0, 0.5)
        
        plt.suptitle(f'Super-Resolution Comparison ({scale_factors[0]:.0f}x upscaling)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sr_comparison.png'), dpi=150)
        plt.close()
        print(f"  Saved: sr_comparison.png")
        
        # 4. 如果是分割数据，生成标签对比图
        if hr_is_segmented:
            sr_segmented = np.round(sr_np * hr_data.max()).astype(np.uint8)
            bicubic_segmented = np.round(bicubic_np * hr_data.max()).astype(np.uint8)
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # 标签视图
            axes[0, 0].imshow(hr_data[slice_idx], cmap='tab10', vmin=0, vmax=3)
            axes[0, 0].set_title('HR GT Labels')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(bicubic_segmented[slice_idx], cmap='tab10', vmin=0, vmax=3)
            axes[0, 1].set_title('Bicubic Labels')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(sr_segmented[slice_idx], cmap='tab10', vmin=0, vmax=3)
            axes[0, 2].set_title('INR Labels')
            axes[0, 2].axis('off')
            
            # 误差视图
            bicubic_label_error = (bicubic_segmented[slice_idx] != hr_data[slice_idx]).astype(float)
            inr_label_error = (sr_segmented[slice_idx] != hr_data[slice_idx]).astype(float)
            
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(bicubic_label_error, cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Bicubic Error ({bicubic_label_error.mean()*100:.1f}%)')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(inr_label_error, cmap='Reds', vmin=0, vmax=1)
            axes[1, 2].set_title(f'INR Error ({inr_label_error.mean()*100:.1f}%)')
            axes[1, 2].axis('off')
            
            plt.suptitle('Label Comparison (Segmented Data)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'label_comparison.png'), dpi=150)
            plt.close()
            print(f"  Saved: label_comparison.png")
        
        # 计算 Bicubic 基线指标
        bicubic_metrics = compute_metrics(bicubic_np, hr_norm)
        print(f"\nBicubic baseline - PSNR: {bicubic_metrics['psnr']:.2f} dB, SSIM: {bicubic_metrics['ssim']:.4f}")
        
        del sr_np, bicubic_np
    
    results = {
        'lr_shape': list(lr_data.shape),
        'hr_shape': list(hr_data.shape),
        'scale_factors': scale_factors,
        'lr_size_mb': float(lr_size / 1024 / 1024),
        'hr_size_mb': float(hr_size / 1024 / 1024),
        'model_size_mb': float(model_size / 1024 / 1024),
        'best_metric': float(best_psnr),
        'training_time_seconds': float(training_time),
        'epochs': int(epochs),
        'final_metrics': metrics,
        'bicubic_metrics': bicubic_metrics,
        'train_losses': [float(l) for l in train_losses],
    }
    
    with open(os.path.join(output_dir, 'sr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Visualizations: loss_curve.png, sr_comparison.png")
    if hr_is_segmented:
        print(f"              label_comparison.png")
    return results


def segment_volume(volume, n_classes=3, max_label=3, lr_is_segmented=False):
    """
    对体积进行分割
    
    Args:
        volume: 输入体积（归一化后的值）
        n_classes: 类别数
        max_label: 最大标签值
        lr_is_segmented: LR 数据是否也是分割数据
    Returns:
        segmented: 分割后的体积
    """
    if lr_is_segmented:
        # LR 是分割数据，直接反归一化并四舍五入
        segmented = np.round(volume * max_label).astype(np.uint8)
        # 确保标签在有效范围内
        segmented = np.clip(segmented, 1, int(max_label))
    else:
        # LR 是灰度数据，使用分位数阈值
        percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        thresholds = np.percentile(volume, percentiles)
        
        segmented = np.zeros_like(volume, dtype=np.uint8)
        for i, thresh in enumerate(thresholds):
            segmented[volume > thresh] = i + 1
        segmented[volume > thresholds[-1]] = n_classes
        segmented = segmented + 1
    
    return segmented


def main():
    parser = argparse.ArgumentParser(description='Super-Resolution Training')
    parser.add_argument('--lr_data', type=str, required=True, help='Low resolution data path')
    parser.add_argument('--hr_data', type=str, required=True, help='High resolution ground truth path')
    parser.add_argument('--output', type=str, default='output/sr', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2**20, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_levels', type=int, default=16, help='Hash encoding levels')
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help='Log2 hashmap size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='MLP hidden dimension')
    parser.add_argument('--validate_every', type=int, default=25, help='Validation frequency')
    
    args = parser.parse_args()
    
    train_sr(
        lr_data_path=args.lr_data,
        hr_data_path=args.hr_data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        hidden_dim=args.hidden_dim,
        validate_every=args.validate_every
    )


if __name__ == '__main__':
    main()
