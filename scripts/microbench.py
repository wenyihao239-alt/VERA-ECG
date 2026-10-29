#!/usr/bin/env python3
"""Micro-benchmark for 3x RTX 3090 planning.

Measures throughput and peak memory for:
  - compact 1D ResNet on 12x1000 ECG
  - ResNet-18 on 3x224x224 ECG images
  - cached-feature fusion MLP
"""
import argparse
import time

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = x if self.down is None else self.down(x)
        return self.act(y + x)


class ResNet1d(nn.Module):
    def __init__(self, in_ch=12, num_classes=5, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers = []
        prev = widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(BasicBlock1d(prev, w, stride=2 if (j == 0 and i > 0) else 1))
                prev = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.body(self.stem(x)).mean(dim=-1)
        return self.head(x)


def bench(model, x, y, device, steps=50, warmup=10, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return steps / dt, x.shape[0] * steps / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)
    x1 = torch.randn(128, 12, 1000, device=device)
    y1 = torch.rand(128, 5, device=device)
    m1 = ResNet1d()
    its, sps, peak = bench(m1, x1, y1, device)
    print(f"signal_resnet1d: {its:.1f} it/s, {sps:.0f} samples/s, peak={peak:.2f} GB, params={sum(p.numel() for p in m1.parameters())/1e6:.1f}M")

    try:
        import torchvision
        m2 = torchvision.models.resnet18(weights=None, num_classes=5)
    except Exception as e:
        print("torchvision unavailable:", e)
        return
    x2 = torch.randn(64, 3, 224, 224, device=device)
    y2 = torch.rand(64, 5, device=device)
    its, sps, peak = bench(m2, x2, y2, device)
    print(f"image_resnet18: {its:.1f} it/s, {sps:.0f} samples/s, peak={peak:.2f} GB, params={sum(p.numel() for p in m2.parameters())/1e6:.1f}M")

    m3 = nn.Sequential(nn.Linear(512 + 512 + 16, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 5))
    x3 = torch.randn(512, 512 + 512 + 16, device=device)
    y3 = torch.rand(512, 5, device=device)
    its, sps, peak = bench(m3, x3, y3, device, steps=200)
    print(f"fusion_mlp: {its:.1f} it/s, {sps:.0f} samples/s, peak={peak:.2f} GB")


if __name__ == "__main__":
    main()
