"""
Avizo/Amira AM 文件读取器
支持 HxByteRLE 压缩格式
"""

import numpy as np
import struct
import re
from typing import Tuple, Optional


def read_am_file(filepath: str) -> np.ndarray:
    """
    读取 Avizo/Amira .am 文件
    
    Args:
        filepath: AM 文件路径
    
    Returns:
        volume: (D, H, W) uint8 体素数据
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # 找到数据段标记 "@1\n"（独立的一行）
    # 注意：头部中的 "@1(HxByteRLE,...)" 不是数据开始
    data_marker = b'@1\n'
    data_start = content.rfind(data_marker)  # 使用 rfind 找最后一个
    
    if data_start == -1:
        # 尝试其他格式
        data_marker = b'@1\r\n'
        data_start = content.rfind(data_marker)
    
    if data_start == -1:
        raise ValueError("无法找到数据段标记 @1")
    
    # 头部是数据标记之前的部分
    header_bytes = content[:data_start]
    header_text = header_bytes.decode('ascii', errors='ignore')
    
    # 数据从标记之后开始
    raw_data = content[data_start + len(data_marker):]
    
    # 解析维度
    dimensions = parse_dimensions(header_text)
    if dimensions is None:
        raise ValueError("无法从头部解析维度信息")
    
    nx, ny, nz = dimensions
    print(f"AM 文件维度: {nx} x {ny} x {nz}")
    
    # 检查压缩类型
    is_rle = 'HxByteRLE' in header_text or 'ByteRLE' in header_text
    
    # 解析 RLE 压缩数据大小
    rle_size = None
    if is_rle:
        rle_match = re.search(r'@1\s*\(HxByteRLE,(\d+)\)', header_text)
        if rle_match:
            rle_size = int(rle_match.group(1))
    
    total_voxels = nx * ny * nz
    
    if is_rle and rle_size:
        print(f"检测到 RLE 压缩，压缩大小: {rle_size} bytes")
        print(f"实际数据大小: {len(raw_data)} bytes")
        volume = decode_hx_byte_rle(raw_data[:rle_size], total_voxels)
    elif is_rle:
        print("检测到 RLE 压缩（未知大小）")
        volume = decode_hx_byte_rle(raw_data, total_voxels)
    else:
        print("检测到原始数据格式")
        volume = np.frombuffer(raw_data[:total_voxels], dtype=np.uint8)
    
    # Reshape
    volume = volume.reshape((nz, ny, nx))
    
    return volume


def parse_dimensions(header: str) -> Optional[Tuple[int, int, int]]:
    """
    从 AM 文件头部解析维度
    
    支持的格式:
    - define Lattice X Y Z
    - Lattice { X Y Z }
    """
    # 格式 1: define Lattice X Y Z
    match = re.search(r'define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)', header, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    # 格式 2: 其他可能的格式
    match = re.search(r'Lattice.*?(\d+)\s+(\d+)\s+(\d+)', header)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    # 格式 3: nx, ny, nz 分别定义
    nx_match = re.search(r'nx\s*[=:]\s*(\d+)', header, re.IGNORECASE)
    ny_match = re.search(r'ny\s*[=:]\s*(\d+)', header, re.IGNORECASE)
    nz_match = re.search(r'nz\s*[=:]\s*(\d+)', header, re.IGNORECASE)
    
    if nx_match and ny_match and nz_match:
        return int(nx_match.group(1)), int(ny_match.group(1)), int(nz_match.group(1))
    
    return None


def decode_hx_byte_rle(data: bytes, expected_size: int) -> np.ndarray:
    """
    解码 HxByteRLE 压缩数据
    
    HxByteRLE 编码规则:
    - 如果第一个字节 >= 128: 重复 (byte - 128) 次下一个字节
    - 如果第一个字节 < 128: 直接复制接下来的 byte 个字节
    
    Args:
        data: 压缩数据
        expected_size: 期望解压后的大小
    
    Returns:
        解压后的数据
    """
    result = []
    pos = 0
    data_len = len(data)
    
    while pos < data_len and len(result) < expected_size:
        count_byte = data[pos]
        pos += 1
        
        if count_byte >= 128:
            # RLE: 重复模式
            repeat_count = count_byte - 128 + 1
            if pos < data_len:
                value = data[pos]
                pos += 1
                result.extend([value] * repeat_count)
        else:
            # 直接复制模式
            copy_count = count_byte + 1
            end_pos = min(pos + copy_count, data_len)
            result.extend(data[pos:end_pos])
            pos = end_pos
    
    # 截断到期望大小
    result = result[:expected_size]
    
    if len(result) < expected_size:
        print(f"警告: 解压数据不完整 ({len(result)}/{expected_size})")
        # 填充
        result.extend([0] * (expected_size - len(result)))
    
    return np.array(result, dtype=np.uint8)


def write_am_file(filepath: str, volume: np.ndarray, compress: bool = True):
    """
    写入 Avizo AM 文件
    
    Args:
        filepath: 输出路径
        volume: (D, H, W) 体素数据
        compress: 是否使用 RLE 压缩
    """
    nz, ny, nx = volume.shape
    volume_flat = volume.astype(np.uint8).flatten()
    
    if compress:
        compressed = encode_hx_byte_rle(volume_flat)
        data_section = f"@1(HxByteRLE,{len(compressed)})\n"
        data_bytes = bytes(compressed)
    else:
        data_section = "@1\n"
        data_bytes = volume_flat.tobytes()
    
    header = f"""# AmiraMesh BINARY-LITTLE-ENDIAN 2.1

define Lattice {nx} {ny} {nz}

Parameters {{
    Content "{nx}x{ny}x{nz} byte, uniform coordinates",
    BoundingBox 0 {nx-1} 0 {ny-1} 0 {nz-1},
    CoordType "uniform"
}}

Lattice {{ byte Data }} = {data_section}"""
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data_bytes)
    
    print(f"AM 文件已保存: {filepath}")


def encode_hx_byte_rle(data: np.ndarray) -> list:
    """
    编码为 HxByteRLE 格式
    
    Args:
        data: 原始数据
    
    Returns:
        压缩后的字节列表
    """
    result = []
    pos = 0
    data_len = len(data)
    
    while pos < data_len:
        # 检查重复
        value = data[pos]
        run_length = 1
        
        while pos + run_length < data_len and run_length < 127:
            if data[pos + run_length] == value:
                run_length += 1
            else:
                break
        
        if run_length >= 3:
            # 使用 RLE
            result.append(128 + run_length - 1)
            result.append(int(value))
            pos += run_length
        else:
            # 查找非重复序列
            literal_start = pos
            literal_len = 0
            
            while pos + literal_len < data_len and literal_len < 127:
                # 检查接下来是否有长重复序列
                if pos + literal_len + 2 < data_len:
                    if (data[pos + literal_len] == data[pos + literal_len + 1] == 
                        data[pos + literal_len + 2]):
                        break
                literal_len += 1
            
            if literal_len > 0:
                result.append(literal_len - 1)
                result.extend([int(x) for x in data[pos:pos + literal_len]])
                pos += literal_len
            else:
                # 避免死循环
                result.append(0)
                result.append(int(data[pos]))
                pos += 1
    
    return result


def create_synthetic_rock(
    shape: Tuple[int, int, int] = (128, 128, 128),
    porosity: float = 0.2,
    pore_size_range: Tuple[int, int] = (3, 15),
    seed: int = 42,
) -> np.ndarray:
    """
    创建合成数字岩心数据（用于测试）
    
    Args:
        shape: 体素尺寸 (D, H, W)
        porosity: 目标孔隙度
        pore_size_range: 孔隙尺寸范围
        seed: 随机种子
    
    Returns:
        volume: 合成岩心数据 (0=孔隙, 255=固体)
    """
    np.random.seed(seed)
    
    D, H, W = shape
    # 从全固体开始
    volume = np.ones(shape, dtype=np.uint8) * 255
    
    total_voxels = D * H * W
    target_pore_voxels = int(total_voxels * porosity)
    current_pore_voxels = 0
    
    min_size, max_size = pore_size_range
    
    while current_pore_voxels < target_pore_voxels:
        # 随机孔隙中心
        z = np.random.randint(min_size, D - min_size)
        y = np.random.randint(min_size, H - min_size)
        x = np.random.randint(min_size, W - min_size)
        
        # 随机椭球孔隙
        rz = np.random.randint(min_size, max_size)
        ry = np.random.randint(min_size, max_size)
        rx = np.random.randint(min_size, max_size)
        
        # 创建椭球
        zz, yy, xx = np.ogrid[
            max(0, z-rz):min(D, z+rz+1),
            max(0, y-ry):min(H, y+ry+1),
            max(0, x-rx):min(W, x+rx+1)
        ]
        
        # 椭球方程
        mask = ((zz - z)**2 / rz**2 + 
                (yy - y)**2 / ry**2 + 
                (xx - x)**2 / rx**2) <= 1
        
        # 设置为孔隙
        volume[max(0, z-rz):min(D, z+rz+1),
               max(0, y-ry):min(H, y+ry+1),
               max(0, x-rx):min(W, x+rx+1)][mask] = 0
        
        current_pore_voxels = (volume == 0).sum()
    
    actual_porosity = current_pore_voxels / total_voxels
    print(f"合成岩心: shape={shape}, 目标孔隙度={porosity:.2%}, 实际孔隙度={actual_porosity:.2%}")
    
    return volume


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 读取指定文件
        filepath = sys.argv[1]
        volume = read_am_file(filepath)
        print(f"体积形状: {volume.shape}")
        print(f"值范围: [{volume.min()}, {volume.max()}]")
        print(f"孔隙度 (阈值=128): {(volume < 128).sum() / volume.size:.4f}")
    else:
        # 创建测试数据
        print("创建合成测试数据...")
        volume = create_synthetic_rock(shape=(64, 64, 64), porosity=0.2)
        
        # 保存为 npy
        np.save('test_rock.npy', volume)
        print("保存为 test_rock.npy")
        
        # 保存为 AM
        write_am_file('test_rock.am', volume, compress=True)
