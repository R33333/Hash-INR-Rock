import numpy as np
import torch
from pathlib import Path

from dataset import load_digital_rock
from model import DigitalRockINR


def compute_toy_porosity(volume: np.ndarray) -> float:
    labels = volume.astype(np.int32)
    return float(np.mean(np.isin(labels, [1, 3])))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data_path = root / "test_data" / "toy_hr_seg.npy"

    volume = load_digital_rock(str(data_path))
    print(f"Loaded test volume: {volume.shape}, dtype={volume.dtype}")
    print(f"Unique labels: {np.unique(volume)}")
    print(f"Toy porosity: {compute_toy_porosity(volume):.4f}")

    # Small CPU smoke test for the model
    model = DigitalRockINR(
        encoding_type='hash',
        n_levels=4,
        log2_hashmap_size=12,
        hidden_dim=16,
        n_hidden_layers=2,
        finest_resolution=max(volume.shape),
    )
    model.eval()

    coords = torch.rand(128, 3)
    with torch.no_grad():
        pred = model(coords)

    print(f"Model forward pass OK: input={tuple(coords.shape)}, output={tuple(pred.shape)}")
    print("Quick test finished successfully.")
