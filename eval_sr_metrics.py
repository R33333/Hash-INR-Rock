"""
超分辨率实验补充评估脚本
为 INR 和 Bicubic 都计算完整的指标（PSNR/SSIM + Accuracy/Porosity）
"""

import torch
import numpy as np
import os
import json
import argparse
from scipy.ndimage import zoom as scipy_zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import DigitalRockINR
from metrics import detect_dataset_type


def evaluate_sr(lr_data_path, hr_data_path, model_dir, n_samples=50_000_000):
    """
    对 INR 和 Bicubic 计算完整的对比指标
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    print("Loading data...")
    lr_data = np.load(lr_data_path)
    hr_data = np.load(hr_data_path)
    
    D_lr, H_lr, W_lr = lr_data.shape
    D_hr, H_hr, W_hr = hr_data.shape
    scale_factors = [D_hr/D_lr, H_hr/H_lr, W_hr/W_lr]
    
    lr_min, lr_max = float(lr_data.min()), float(lr_data.max())
    hr_unique = np.unique(hr_data)
    hr_max_val = hr_data.max()
    
    print(f"LR: {lr_data.shape}, HR: {hr_data.shape}")
    print(f"Scale: {scale_factors[0]:.0f}x, Labels: {hr_unique}")
    
    # 检测数据集类型
    hr_dataset_type = detect_dataset_type(hr_data)
    print(f"Dataset type: {hr_dataset_type}")
    
    if hr_dataset_type == 'mrccm':
        pore_labels = [1, 3]
    else:
        pore_labels = [hr_unique.min()]
    
    # ========== 1. 加载 INR 模型 ==========
    print("\n" + "="*50)
    print("Loading INR model...")
    model = DigitalRockINR(
        n_levels=16,
        log2_hashmap_size=19,
        hidden_dim=64
    ).to(device)
    
    model_path = os.path.join(model_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}")
    
    # ========== 2. 采样 HR 坐标 ==========
    print(f"\nSampling {n_samples//1_000_000}M points at HR resolution...")
    
    z_idx = torch.randint(0, D_hr, (n_samples,))
    y_idx = torch.randint(0, H_hr, (n_samples,))
    x_idx = torch.randint(0, W_hr, (n_samples,))
    
    # HR ground truth（整数标签）
    hr_samples = hr_data[z_idx.numpy(), y_idx.numpy(), x_idx.numpy()]
    # HR ground truth 归一化（用于 PSNR/SSIM 计算）
    hr_samples_norm = hr_samples.astype(np.float32) / hr_max_val
    
    # ========== 3. INR 预测 ==========
    print("INR inference...")
    x_norm = x_idx.float() / (W_hr - 1)
    y_norm = y_idx.float() / (H_hr - 1)
    z_norm = z_idx.float() / (D_hr - 1)
    
    batch_infer = 2_000_000
    inr_samples = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_infer):
            end = min(i + batch_infer, n_samples)
            coords = torch.stack([
                x_norm[i:end], y_norm[i:end], z_norm[i:end]
            ], dim=-1).to(device)
            pred = model(coords).squeeze(-1).cpu()
            inr_samples.append(pred)
    
    # INR 输出（归一化 [0,1]）
    inr_samples = torch.cat(inr_samples).numpy()
    # 反归一化到原始标签范围
    inr_denorm = inr_samples * (lr_max - lr_min) + lr_min
    # INR 归一化到 [0,1]（用于 PSNR/SSIM）
    inr_norm = inr_denorm / hr_max_val
    # INR 分割标签
    inr_labels = np.round(inr_denorm).astype(np.uint8)
    inr_labels = np.clip(inr_labels, 1, int(hr_max_val))
    
    # ========== 4. Bicubic 预测 ==========
    print("Bicubic interpolation...")
    # 对采样点计算对应的 LR 坐标并用 bicubic 插值
    # 使用 scipy_zoom 对 LR 数据做整体 bicubic 上采样太大，改用采样方式
    # 方法：对每个采样点，找到其在 LR 中的浮点坐标，用 scipy map_coordinates 插值
    from scipy.ndimage import map_coordinates
    
    # HR 坐标映射到 LR 坐标空间
    lr_z = z_idx.numpy().astype(np.float64) * (D_lr - 1) / (D_hr - 1)
    lr_y = y_idx.numpy().astype(np.float64) * (H_lr - 1) / (H_hr - 1)
    lr_x = x_idx.numpy().astype(np.float64) * (W_lr - 1) / (W_hr - 1)
    
    # 分批插值（map_coordinates 内存消耗大）
    batch_size = 5_000_000
    bicubic_samples = []
    lr_float = lr_data.astype(np.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        coords_lr = np.array([lr_z[i:end], lr_y[i:end], lr_x[i:end]])
        interp = map_coordinates(lr_float, coords_lr, order=3, mode='nearest')
        bicubic_samples.append(interp.astype(np.float32))
        if (i // batch_size) % 2 == 0:
            print(f"  Bicubic: {end}/{n_samples} ({100*end/n_samples:.0f}%)")
    
    bicubic_samples = np.concatenate(bicubic_samples)
    # Bicubic 归一化（用于 PSNR/SSIM）
    bicubic_norm = bicubic_samples / hr_max_val
    bicubic_norm = np.clip(bicubic_norm, 0, 1)
    # Bicubic 分割标签
    bicubic_labels = np.round(bicubic_samples).astype(np.uint8)
    bicubic_labels = np.clip(bicubic_labels, 1, int(hr_max_val))
    
    # Nearest Neighbor 基线 (order=0)
    print("Nearest neighbor interpolation...")
    nn_samples = map_coordinates(lr_float, 
                                  np.array([lr_z, lr_y, lr_x]), 
                                  order=0, mode='nearest').astype(np.float32)
    nn_norm = nn_samples / hr_max_val
    nn_labels = np.round(nn_samples).astype(np.uint8)
    nn_labels = np.clip(nn_labels, 1, int(hr_max_val))
    
    # Trilinear 基线 (order=1)
    print("Trilinear interpolation...")
    trilinear_samples = []
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        coords_lr = np.array([lr_z[i:end], lr_y[i:end], lr_x[i:end]])
        interp = map_coordinates(lr_float, coords_lr, order=1, mode='nearest')
        trilinear_samples.append(interp.astype(np.float32))
    trilinear_samples = np.concatenate(trilinear_samples)
    trilinear_norm = trilinear_samples / hr_max_val
    trilinear_norm = np.clip(trilinear_norm, 0, 1)
    trilinear_labels = np.round(trilinear_samples).astype(np.uint8)
    trilinear_labels = np.clip(trilinear_labels, 1, int(hr_max_val))
    
    del lr_float
    
    # ========== 5. 计算所有指标 ==========
    print("\n" + "="*50)
    print("Computing metrics...")
    print("="*50)
    
    results = {}
    
    for method_name, pred_norm, pred_labels in [
        ('Nearest', nn_norm, nn_labels),
        ('Trilinear', trilinear_norm, trilinear_labels),
        ('Bicubic', bicubic_norm, bicubic_labels),
        ('INR', inr_norm, inr_labels),
    ]:
        print(f"\n--- {method_name} ---")
        
        # PSNR
        psnr = peak_signal_noise_ratio(
            hr_samples_norm, pred_norm.astype(np.float32), 
            data_range=1.0
        )
        print(f"  PSNR: {psnr:.2f} dB")
        
        # MAE
        mae = np.mean(np.abs(hr_samples_norm - pred_norm))
        print(f"  MAE: {mae:.4f}")
        
        # Accuracy（总体）
        accuracy = np.mean(pred_labels == hr_samples)
        print(f"  Segmentation Accuracy: {accuracy*100:.2f}%")
        
        # 各类别准确率
        label_accs = {}
        for label in hr_unique:
            mask = hr_samples == label
            if mask.sum() > 0:
                label_acc = np.mean(pred_labels[mask] == label)
                label_accs[int(label)] = float(label_acc)
                print(f"    Label {label}: {label_acc*100:.2f}%")
        
        # 孔隙度
        porosity_gt = np.mean(np.isin(hr_samples, pore_labels))
        porosity_pred = np.mean(np.isin(pred_labels, pore_labels))
        porosity_error = abs(porosity_gt - porosity_pred)
        print(f"  Porosity - GT: {porosity_gt:.4f}, Pred: {porosity_pred:.4f}, Error: {porosity_error:.4f}")
        
        results[method_name] = {
            'psnr': float(psnr),
            'mae': float(mae),
            'accuracy': float(accuracy),
            'label_accuracy': label_accs,
            'porosity_gt': float(porosity_gt),
            'porosity_pred': float(porosity_pred),
            'porosity_error': float(porosity_error),
        }
    
    # ========== 6. 打印对比表格 ==========
    print("\n" + "="*60)
    print(f"{'Method':<12} {'PSNR':>8} {'MAE':>8} {'Acc%':>8} {'Poro.Err':>10}")
    print("-"*60)
    for method in ['Nearest', 'Trilinear', 'Bicubic', 'INR']:
        r = results[method]
        print(f"{method:<12} {r['psnr']:>8.2f} {r['mae']:>8.4f} {r['accuracy']*100:>8.2f} {r['porosity_error']:>10.4f}")
    print("="*60)
    
    # 保存结果
    output_path = os.path.join(model_dir, 'full_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SR Full Metrics Evaluation')
    parser.add_argument('--lr_data', type=str, required=True)
    parser.add_argument('--hr_data', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=50_000_000)
    
    args = parser.parse_args()
    evaluate_sr(args.lr_data, args.hr_data, args.model_dir, args.n_samples)


if __name__ == '__main__':
    main()
