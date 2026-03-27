# INR for Digital Rock Compression

Code for the paper: **Hash-Encoded Implicit Neural Representations for Efficient Compression of Large-Scale Digital Rock CT Images** 

A compact neural network is trained to map spatial coordinates to voxel labels. The trained weights serve as the compressed file. Because the mapping is continuous, the same model supports arbitrary-resolution queries (super-resolution) at no additional training cost.

## Results

| Config | Model Size | Compression Ratio | Label Accuracy | Porosity Error |
|--------|-----------|-------------------|----------------|----------------|
| Tiny   | 16 MB     | 976×              | 90.9%          | 4.69%          |
| Small  | 32 MB     | 489×              | 92.4%          | 3.49%          |
| Medium | 128 MB    | 122×              | 94.8%          | 1.94%          |
| Large  | 256 MB    | 61×               | 95.7%          | 1.42%          |

Dataset: ILS carbonate rock, 15.29 GB original size.

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Recommended: NVIDIA GPU with 24 GB VRAM. For large datasets (>15 GB), 64 GB+ RAM is required.

## Usage

### Compression

```bash
python train_fast.py \
    --data data/ils_hr_seg.npy \
    --epochs 100 \
    --batch_size 1048576 \
    --log2_hashmap_size 20 \
    --output output/ils_compression
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | — | Path to `.npy` segmented volume |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 1048576 | Coordinates per batch |
| `--log2_hashmap_size` | 20 | Hash table size (controls compression ratio) |
| `--output` | `output/` | Output directory |

Increase `--log2_hashmap_size` for higher accuracy (lower compression); decrease for higher compression.

### Super-Resolution

```bash
python train_sr.py \
    --lr_data data/ils_lr_seg_4x.npy \
    --hr_data data/ils_hr_seg.npy \
    --epochs 50 \
    --batch_size 1048576 \
    --output output/ils_sr_4x
```

### Evaluate SR Metrics

```bash
python eval_sr_metrics.py \
    --lr_data data/ils_lr_seg_4x.npy \
    --hr_data data/ils_hr_seg.npy \
    --model_dir output/ils_sr_4x
```

## File Structure

```
├── model.py              # Hash encoding + MLP architecture
├── train_fast.py         # Compression training (large dataset support)
├── train_sr.py           # Super-resolution training
├── eval_sr_metrics.py    # SR evaluation against baselines
├── metrics.py            # PSNR, SSIM, porosity calculation
├── dataset.py            # Data loading (supports .npy, .am, .mat)
├── am_loader.py          # Avizo AM format reader
├── preprocess_data.py    # Data preprocessing utilities
└── requirements.txt      # Dependencies
```

## Data

Experiments use the [MRCCM dataset](https://www.doi.org/10.17612/3t36-q704) (Alqahtani et al., 2021) from the Digital Porous Media Portal, which includes the Indiana Limestone (ILS) and Middle Eastern Carbonate (MEC) samples.

## Citation

Coming soon.
