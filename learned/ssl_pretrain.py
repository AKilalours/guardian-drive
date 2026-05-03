"""
learned/ssl_pretrain.py
Guardian Drive -- Self-Supervised Contrastive Pretraining

Implements SimCLR-style contrastive learning on WESAD physiological
sequences WITHOUT labels. The encoder learns representations by
pulling augmented views of the same window together and pushing
different windows apart.

This is genuine self-supervised learning:
- No labels used during pretraining
- Augmentations: Gaussian noise, time shift, channel dropout, scaling
- Loss: NT-Xent (normalized temperature-scaled cross entropy)
- Pretrained encoder then fine-tuned for low-arousal classification

Why this matters for Tesla:
- Fleet data has no labels -- SSL enables learning from unlabeled sensor streams
- Same paradigm Tesla uses for vision SSL on unlabeled dashcam footage

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ── Augmentations ─────────────────────────────────────────────────
class PhysioAugment:
    """
    Augmentation pipeline for physiological sequences.
    Two views of same window should produce similar representations.
    """
    def __init__(self, noise_std=0.05, scale_range=(0.8,1.2),
                 shift_max=50, dropout_p=0.1):
        self.noise_std   = noise_std
        self.scale_range = scale_range
        self.shift_max   = shift_max
        self.dropout_p   = dropout_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [C, T]"""
        # Random Gaussian noise
        x = x + torch.randn_like(x) * self.noise_std

        # Random amplitude scaling per channel
        scale = torch.FloatTensor(x.shape[0], 1).uniform_(*self.scale_range)
        x = x * scale

        # Random time shift
        shift = np.random.randint(-self.shift_max, self.shift_max)
        x = torch.roll(x, shift, dims=1)

        # Random channel dropout
        if np.random.random() < self.dropout_p:
            ch = np.random.randint(x.shape[0])
            x[ch] = x[ch] * 0

        return x


# ── Encoder (shared with TCN) ──────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = (3-1)*dilation
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out[:,:,:x.size(2)] + self.res(x)

class PhysioEncoder(nn.Module):
    """TCN encoder -- same architecture as supervised model."""
    def __init__(self, d_out=128):
        super().__init__()
        self.b1   = TCNBlock(4,  32, dilation=1)
        self.b2   = TCNBlock(32, 64, dilation=2)
        self.b3   = TCNBlock(64, 64, dilation=4)
        self.b4   = TCNBlock(64, 64, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d_out))
    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(self.pool(x).squeeze(-1))


# ── NT-Xent Loss (SimCLR) ─────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Chen et al., 2020 (SimCLR).
    Pulls augmented views of same sample together,
    pushes all other samples apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)          # [2B, D]
        sim = (z @ z.T) / self.T                  # [2B, 2B]

        # Mask self-similarity
        mask = torch.eye(2*B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B),
                            torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, labels)


# ── SSL Training ──────────────────────────────────────────────────
def pretrain_ssl(data_dir: str = "datasets/WESAD/WESAD",
                 epochs: int = 20,
                 batch_size: int = 64,
                 lr: float = 3e-4,
                 save_path: str = "learned/models/wesad_ssl_encoder.pt"):
    """
    SSL pretraining on WESAD without labels.
    Uses all windows from all subjects -- no train/test split needed
    because no labels are used.
    """
    import pickle
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SSL pretraining on: {device}")
    print(f"No labels used -- self-supervised learning")

    # Load all WESAD windows (no label filtering)
    data_path = Path(data_dir)
    windows = []
    WIN, STEP = 4200, 2100

    if data_path.exists():
        for subj_dir in sorted(data_path.iterdir()):
            pkl = subj_dir / f"{subj_dir.name}.pkl"
            if not pkl.exists(): continue
            with open(pkl, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            sig = data["signal"]["chest"]
            ecg = sig["ECG"].flatten()
            eda = sig["EDA"].flatten()
            tmp = sig["Temp"].flatten()
            rsp = sig["Resp"].flatten()
            n = min(len(ecg), len(eda), len(tmp), len(rsp))
            for s in range(0, n-WIN, STEP):
                w = np.stack([ecg[s:s+WIN], eda[s:s+WIN],
                              tmp[s:s+WIN], rsp[s:s+WIN]])
                windows.append(w.astype(np.float32))
        print(f"Loaded {len(windows)} windows (no labels)")
    else:
        # Demo mode without data
        print("WESAD data not found -- running demo with random windows")
        windows = [np.random.randn(4, 4200).astype(np.float32)
                   for _ in range(500)]

    X   = torch.FloatTensor(np.array(windows))
    mu  = X.mean(dim=(0,2), keepdim=True)
    std = X.std(dim=(0,2),  keepdim=True) + 1e-6
    X   = (X - mu) / std

    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, drop_last=True)

    aug     = PhysioAugment()
    encoder = PhysioEncoder(d_out=128).to(device)
    opt     = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = NTXentLoss(temperature=0.07)

    best_loss = float("inf")
    for ep in range(1, epochs+1):
        encoder.train()
        total = 0
        for (batch,) in loader:
            batch = batch.to(device)
            # Two augmented views -- NO labels
            v1 = torch.stack([aug(b) for b in batch])
            v2 = torch.stack([aug(b) for b in batch])
            z1 = encoder(v1)
            z2 = encoder(v2)
            loss = loss_fn(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        saved = ""
        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), save_path)
            saved = "saved"
        print(f"  SSL ep {ep:02d}/{epochs}  loss={avg:.4f}  {saved}")

    print(f"\nSSL pretraining done. Best loss={best_loss:.4f}")
    print(f"Encoder saved: {save_path}")
    print("Use this encoder as backbone for Task B fine-tuning")
    return save_path


if __name__ == "__main__":
    pretrain_ssl(epochs=5)  # quick demo -- use 20 for full run
