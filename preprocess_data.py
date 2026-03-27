"""
数据预处理脚本
用于检查、验证和预处理数字岩心数据

功能:
1. 验证 AM 文件读取
2. 显示数据统计信息
3. 可视化预览
4. 转换为 npy 格式（可选）

使用方法:
    # 检查数据
    python preprocess_data.py --data path/to/data.am --info
    
    # 可视化预览
    python preprocess_data.py --data path/to/data.am --visualize
    
    # 转换为 npy
    python preprocess_data.py --data path/to/data.am --convert
    
    # 全部操作
    python preprocess_data.py --data path/to/data.am --all
"""

import os
import argparse
import numpy as np
from am_loader import read_am_file, create_synthetic_rock


def check_data_info(data_path: str):
    """检查数据信息"""
    print(f"\n{'='*60}")
    print("DATA INFORMATION")
    print(f"{'='*60}")
    
    print(f"\nFile: {data_path}")
    print(f"File size: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")
    
    # 读取数据
    print("\nLoading data...")
    volume = read_am_file(data_path)
    
    print(f"\n--- Volume Statistics ---")
    print(f"Shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Memory size: {volume.nbytes / 1024 / 1024:.2f} MB")
    print(f"Value range: [{volume.min()}, {volume.max()}]")
    
    # 唯一值分析
    unique_values = np.unique(volume)
    print(f"\n--- Value Distribution ---")
    print(f"Number of unique values: {len(unique_values)}")
    
    if len(unique_values) <= 20:
        print(f"Unique values: {unique_values}")
        print("\nValue counts:")
        for val in unique_values:
            count = (volume == val).sum()
            percentage = count / volume.size * 100
            print(f"  Label {val}: {count:,} voxels ({percentage:.2f}%)")
    else:
        print("(Continuous data detected)")
        print(f"Mean: {volume.mean():.4f}")
        print(f"Std: {volume.std():.4f}")
    
    # 孔隙度估计（假设 label 0 是孔隙）
    print(f"\n--- Porosity Estimate ---")
    if len(unique_values) <= 20:
        # 标签数据
        pore_count = (volume == 0).sum()
        porosity = pore_count / volume.size
        print(f"Porosity (label=0): {porosity:.4f} ({porosity*100:.2f}%)")
    else:
        # 灰度数据
        threshold = 128 if volume.max() > 1 else 0.5
        pore_count = (volume < threshold).sum()
        porosity = pore_count / volume.size
        print(f"Porosity (threshold={threshold}): {porosity:.4f} ({porosity*100:.2f}%)")
    
    print(f"\n{'='*60}\n")
    
    return volume


def visualize_data(volume: np.ndarray, save_dir: str = None):
    """可视化数据切片"""
    import matplotlib.pyplot as plt
    
    print("Generating visualization...")
    
    D, H, W = volume.shape
    
    # 归一化用于显示
    if volume.max() > 1:
        volume_display = volume / volume.max()
    else:
        volume_display = volume
    
    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 三个正交切片
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2
    
    axes[0, 0].imshow(volume_display[z_mid, :, :], cmap='gray')
    axes[0, 0].set_title(f'XY Plane (z={z_mid})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(volume_display[:, y_mid, :], cmap='gray')
    axes[0, 1].set_title(f'XZ Plane (y={y_mid})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(volume_display[:, :, x_mid], cmap='gray')
    axes[0, 2].set_title(f'YZ Plane (x={x_mid})')
    axes[0, 2].axis('off')
    
    # 值分布直方图
    axes[1, 0].hist(volume.flatten(), bins=50, color='steelblue', edgecolor='black')
    axes[1, 0].set_title('Value Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    
    # 多个 Z 切片
    z_positions = [D // 4, D // 2, 3 * D // 4]
    for i, z in enumerate(z_positions[1:], 1):
        axes[1, i].imshow(volume_display[z, :, :], cmap='gray')
        axes[1, i].set_title(f'Z Slice at z={z}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'data_preview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved preview to: {save_path}")
    
    plt.show()
    
    # 如果是标签数据，显示各相分布
    unique_values = np.unique(volume)
    if len(unique_values) <= 10:
        print("\nGenerating phase visualization...")
        
        fig, axes = plt.subplots(1, len(unique_values), figsize=(4 * len(unique_values), 4))
        if len(unique_values) == 1:
            axes = [axes]
        
        z_mid = D // 2
        for i, val in enumerate(unique_values):
            phase_slice = (volume[z_mid] == val).astype(float)
            axes[i].imshow(phase_slice, cmap='binary')
            count = (volume == val).sum()
            pct = count / volume.size * 100
            axes[i].set_title(f'Phase {val}\n({pct:.1f}%)')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'phase_distribution.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved phase distribution to: {save_path}")
        
        plt.show()


def convert_to_npy(volume: np.ndarray, output_path: str):
    """转换为 npy 格式"""
    print(f"Saving to: {output_path}")
    np.save(output_path, volume)
    
    saved_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved size: {saved_size:.2f} MB")
    
    # 验证
    loaded = np.load(output_path)
    assert np.array_equal(volume, loaded), "Verification failed!"
    print("Verification passed!")


def create_test_data(output_dir: str):
    """创建测试数据"""
    print("Creating synthetic test data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 小尺寸测试数据
    volume_small = create_synthetic_rock(shape=(64, 64, 64), porosity=0.2, seed=42)
    np.save(os.path.join(output_dir, 'test_64.npy'), volume_small)
    print(f"Created: {output_dir}/test_64.npy")
    
    # 中等尺寸测试数据
    volume_medium = create_synthetic_rock(shape=(128, 128, 128), porosity=0.2, seed=42)
    np.save(os.path.join(output_dir, 'test_128.npy'), volume_medium)
    print(f"Created: {output_dir}/test_128.npy")
    
    # 大尺寸测试数据
    volume_large = create_synthetic_rock(shape=(256, 256, 256), porosity=0.2, seed=42)
    np.save(os.path.join(output_dir, 'test_256.npy'), volume_large)
    print(f"Created: {output_dir}/test_256.npy")
    
    print("\nTest data created successfully!")
    return volume_small


def batch_process(data_dir: str, output_dir: str):
    """批量处理目录下的所有 AM 文件"""
    import glob
    
    am_files = glob.glob(os.path.join(data_dir, '**/*.am'), recursive=True)
    
    if not am_files:
        print(f"No AM files found in {data_dir}")
        return
    
    print(f"Found {len(am_files)} AM files")
    os.makedirs(output_dir, exist_ok=True)
    
    for am_path in am_files:
        print(f"\nProcessing: {am_path}")
        try:
            volume = read_am_file(am_path)
            
            # 生成输出文件名
            basename = os.path.splitext(os.path.basename(am_path))[0]
            npy_path = os.path.join(output_dir, f'{basename}.npy')
            
            np.save(npy_path, volume)
            print(f"  -> Saved: {npy_path}")
            print(f"     Shape: {volume.shape}, Size: {volume.nbytes / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"  -> Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Digital Rock Data Preprocessing')
    parser.add_argument('--data', type=str, help='Path to AM or npy file')
    parser.add_argument('--info', action='store_true', help='Show data information')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    parser.add_argument('--convert', action='store_true', help='Convert to npy format')
    parser.add_argument('--output', type=str, default='preprocessed', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Run all operations')
    parser.add_argument('--create_test', action='store_true', help='Create synthetic test data')
    parser.add_argument('--batch', type=str, help='Batch process directory')
    
    args = parser.parse_args()
    
    # 创建测试数据
    if args.create_test:
        create_test_data(args.output)
        return
    
    # 批量处理
    if args.batch:
        batch_process(args.batch, args.output)
        return
    
    # 需要数据路径
    if not args.data:
        parser.print_help()
        print("\n示例:")
        print('  python preprocess_data.py --data "DRP-265/Berea Altered_ mixed-wet/Berea_Altered_waterflooding_segmented/Berea_Altered_waterflooding_segmented.am" --all')
        print('  python preprocess_data.py --create_test --output test_data')
        return
    
    # 检查文件存在
    if not os.path.exists(args.data):
        print(f"Error: File not found: {args.data}")
        return
    
    volume = None
    
    # 数据信息
    if args.info or args.all:
        volume = check_data_info(args.data)
    
    # 可视化
    if args.visualize or args.all:
        if volume is None:
            volume = read_am_file(args.data) if args.data.endswith('.am') else np.load(args.data)
        visualize_data(volume, args.output if args.all else None)
    
    # 转换
    if args.convert or args.all:
        if volume is None:
            volume = read_am_file(args.data) if args.data.endswith('.am') else np.load(args.data)
        
        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.data))[0]
        npy_path = os.path.join(args.output, f'{basename}.npy')
        convert_to_npy(volume, npy_path)


if __name__ == '__main__':
    main()
