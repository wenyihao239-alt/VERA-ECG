#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ECGArray(Dataset):
    def __init__(self, data_dir, split, task, augment="mild", train=False):
        self.x = np.load(os.path.join(data_dir, f"X_{split}.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(data_dir, f"y_{split}_{task}.npy")).astype(np.float32)
        self.augment = augment if train else "none"
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = np.array(self.x[i], dtype=np.float32, copy=True)
        y = self.y[i]
        if self.train:
            x = augment_signal(x, self.augment)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mu) / sd
        return torch.from_numpy(x), torch.from_numpy(y)


def augment_signal(x, mode):
    if mode == "none":
        return x
    if mode == "mild":
        if np.random.rand() < 0.5:
            x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            t = np.arange(x.shape[1], dtype=np.float32) / 100.0
            f = np.random.uniform(0.05, 0.5)
            a = np.random.uniform(0.01, 0.05)
            x = x + (a * np.sin(2 * np.pi * f * t)).astype(np.float32)[None, :]
        if np.random.rand() < 0.3:
            for _ in range(np.random.randint(1, 3)):
                k = np.random.randint(x.shape[0])
                x[k] = 0
        if np.random.rand() < 0.3:
            x = x * np.random.uniform(0.9, 1.1)
    elif mode == "strong":
        if np.random.rand() < 0.8:
            snr = np.random.uniform(5, 20)
            sig_power = np.mean(x ** 2) + 1e-8
            noise_power = sig_power / (10 ** (snr / 10))
            x = x + np.random.normal(0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            t = np.arange(x.shape[1], dtype=np.float32) / 100.0
            f = np.random.uniform(0.05, 0.7)
            a = np.random.uniform(0.05, 0.3)
            x = x + (a * np.sin(2 * np.pi * f * t)).astype(np.float32)[None, :]
        if np.random.rand() < 0.5:
            for _ in range(np.random.randint(1, 4)):
                k = np.random.randint(x.shape[0])
                x[k] = 0
        if np.random.rand() < 0.3:
            x = x * np.random.uniform(0.7, 1.3)
    return x.astype(np.float32)


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
        self.out_dim = prev
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x, return_features=False):
        z = self.body(self.stem(x)).mean(dim=-1)
        if return_features:
            return self.head(z), z
        return self.head(z)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.b0 = nn.Conv1d(out_ch, out_ch, 9, padding=4, bias=False)
        self.b1 = nn.Conv1d(out_ch, out_ch, 19, padding=9, bias=False)
        self.b2 = nn.Conv1d(out_ch, out_ch, 39, padding=19, bias=False)
        self.b3 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(out_ch, out_ch, 1, bias=False))
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        self.res = nn.Sequential(nn.Conv1d(in_ch, out_ch * 4, 1, bias=False), nn.BatchNorm1d(out_ch * 4))

    def forward(self, x):
        y = self.act(self.bottleneck(x))
        y = torch.cat([self.b0(y), self.b1(y), self.b2(y), self.b3(y)], dim=1)
        y = self.act(self.bn(y) + self.res(x))
        return y


class InceptionTime(nn.Module):
    def __init__(self, in_ch=12, num_classes=5, width=32, blocks=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 1, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[InceptionBlock(width if i == 0 else width * 4, width) for i in range(blocks)])
        self.out_dim = width * 4
        self.head = nn.Linear(self.out_dim, num_classes)

    def forward(self, x, return_features=False):
        z = self.body(self.stem(x)).mean(dim=-1)
        if return_features:
            return self.head(z), z
        return self.head(z)


def macro_auc(y, p):
    vals = []
    for j in range(y.shape[1]):
        if y[:, j].sum() > 0 and y[:, j].sum() < len(y):
            vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def macro_ap(y, p):
    vals = []
    for j in range(y.shape[1]):
        if y[:, j].sum() > 0:
            vals.append(average_precision_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y, p, thr=0.5):
    return float(f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0))


def best_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f = macro_f1(y, p, t)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def ece(y, p, n_bins=15):
    y = y.reshape(-1)
    p = p.reshape(-1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(p)) * abs(acc - conf)
    return float(e)


def evaluate(model, loader, device, task):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--model", default="resnet1d", choices=["resnet1d", "inceptiontime"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = ECGArray(args.data_dir, "train", args.task, args.augment, train=True)
    val_ds = ECGArray(args.data_dir, "val", args.task, "none", train=False)
    test_ds = ECGArray(args.data_dir, "test", args.task, "none", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_ds.y.shape[1]
    if args.model == "resnet1d":
        model = ResNet1d(num_classes=num_classes)
    else:
        model = InceptionTime(num_classes=num_classes)
    model.to(device)

    pos = train_ds.y.sum(axis=0)
    neg = len(train_ds.y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.5, 10.0), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight if args.task == "super" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_auc = -1.0
    best_epoch = -1
    bad = 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_loss,val_auc,val_ap,val_f1\n")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        yv, pv = evaluate(model, val_loader, device, args.task)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(pv).clamp(1e-6, 1 - 1e-6), torch.from_numpy(yv)).item())
        auc = macro_auc(yv, pv)
        ap = macro_ap(yv, pv)
        f1 = macro_f1(yv, pv, 0.5)
        lr = opt.param_groups[0]["lr"]
        line = f"{epoch},{lr:.6g},{np.mean(losses):.6f},{val_loss:.6f},{auc:.6f},{ap:.6f},{f1:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_auc": auc}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break
    elapsed = time.time() - t0

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    yv, pv = evaluate(model, val_loader, device, args.task)
    thr, val_f1_best = best_threshold(yv, pv)
    yt, pt = evaluate(model, test_loader, device, args.task)
    metrics = {
        "best_epoch": best_epoch,
        "val_auc": macro_auc(yv, pv),
        "val_ap": macro_ap(yv, pv),
        "val_f1_at_0.5": macro_f1(yv, pv, 0.5),
        "val_f1_best": val_f1_best,
        "threshold": thr,
        "test_auc": macro_auc(yt, pt),
        "test_ap": macro_ap(yt, pt),
        "test_f1_at_0.5": macro_f1(yt, pt, 0.5),
        "test_f1_at_val_thr": macro_f1(yt, pt, thr),
        "test_ece": ece(yt, pt),
        "test_brier": float(np.mean((pt - yt) ** 2)),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "elapsed_sec": elapsed,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out_dir, "test_pred.npy"), pt)
    np.save(os.path.join(args.out_dir, "test_label.npy"), yt)
    np.save(os.path.join(args.out_dir, "val_pred.npy"), pv)
    np.save(os.path.join(args.out_dir, "val_label.npy"), yv)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

