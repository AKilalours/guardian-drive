#!/usr/bin/env python3
"""
Train a lightweight CNN on a REAL labeled dataset.

Dataset format:
  data/vision/
    alert/
      *.jpg
    drowsy/
      *.jpg

This is supervised learning; you need labels (use capture_dataset.py or a public dataset).

Run:
  python -m vision.train_cnn --data data/vision --out runs/vision_model.pt

Notes:
  - This is a starter script. For production you want:
      * more data (public datasets + your own)
      * subject split (no leakage)
      * temporal modeling (video sequence -> LSTM/TCN)
      * calibration (threshold tuning)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/vision")
    p.add_argument("--out", type=str, default="runs/vision_model.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"Dataset path not found: {data_dir}")

    tfm = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(str(data_dir), transform=tfm)
    if len(ds.classes) != 2:
        raise SystemExit(f"Expected 2 classes (alert/drowsy). Got: {ds.classes}")
    print("Classes:", ds.classes, "Samples:", len(ds))

    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # MobileNetV3 small (fast on CPU)
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m = m.to(device)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        m.train()
        tr_loss = 0.0
        tr_ok = 0
        tr_n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = m(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * x.size(0)
            tr_ok += int((logits.argmax(dim=1) == y).sum().item())
            tr_n += x.size(0)

        m.eval()
        va_ok = 0
        va_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = m(x)
                va_ok += int((logits.argmax(dim=1) == y).sum().item())
                va_n += x.size(0)

        tr_acc = tr_ok / max(1, tr_n)
        va_acc = va_ok / max(1, va_n)
        tr_loss = tr_loss / max(1, tr_n)

        print(f"epoch {ep:02d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": m.state_dict(), "classes": ds.classes, "img": args.img}, out_path)
            print(f"  saved: {out_path}  (best_val={best_val:.3f})")

    print("Done. Best val acc:", best_val)


if __name__ == "__main__":
    main()
