#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_signal import (
    ResNet1d, InceptionTime, augment_signal, best_threshold, ece,
    macro_ap, macro_auc, macro_f1, set_seed,
)


class ECGArrayMV(Dataset):
    def __init__(self, data_dir, mv_dir, split, task, augment="mild", train=False, mv_mean=None, mv_std=None):
        self.x = np.load(os.path.join(data_dir, f"X_{split}.npy"), mmap_mode="r")
        self.mv = np.load(os.path.join(mv_dir, f"mv_{split}.npy")).astype(np.float32)
        self.y = np.load(os.path.join(data_dir, f"y_{split}_{task}.npy")).astype(np.float32)
        self.augment = augment if train else "none"
        self.train = train
        self.mv_mean = np.asarray(mv_mean).reshape(-1)
        self.mv_std = np.asarray(mv_std).reshape(-1)

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
        mv = (self.mv[i] - self.mv_mean) / self.mv_std
        return torch.from_numpy(x), torch.from_numpy(mv.astype(np.float32)), torch.from_numpy(y)


class ResNet1dMV(ResNet1d):
    def __init__(self, in_ch=12, num_classes=5, mv_dim=36, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__(in_ch=in_ch, num_classes=num_classes, widths=widths, blocks=blocks)
        self.mv_mlp = nn.Sequential(
            nn.Linear(mv_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * self.out_dim),
        )

    def forward(self, x, mv=None, return_features=False):
        z = self.body(self.stem(x))
        if mv is not None:
            gb = self.mv_mlp(mv.float())
            g, b = gb.chunk(2, dim=1)
            z = z * (1.0 + g[:, :, None]) + b[:, :, None]
        z = z.mean(dim=-1)
        if return_features:
            return self.head(z), z
        return self.head(z)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, mv, y in loader:
            x, mv = x.to(device, non_blocking=True), mv.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x, mv)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mv-dir", required=True)
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

    mv_tr_raw = np.load(os.path.join(args.mv_dir, "mv_train.npy"))
    mv_mean = mv_tr_raw.mean(axis=0, keepdims=True).astype(np.float32)
    mv_std = (mv_tr_raw.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    np.save(os.path.join(args.out_dir, "mv_mean.npy"), mv_mean)
    np.save(os.path.join(args.out_dir, "mv_std.npy"), mv_std)

    train_ds = ECGArrayMV(args.data_dir, args.mv_dir, "train", args.task, args.augment, True, mv_mean, mv_std)
    val_ds = ECGArrayMV(args.data_dir, args.mv_dir, "val", args.task, "none", False, mv_mean, mv_std)
    test_ds = ECGArrayMV(args.data_dir, args.mv_dir, "test", args.task, "none", False, mv_mean, mv_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_ds.y.shape[1]
    model = ResNet1dMV(num_classes=num_classes, mv_dim=mv_tr_raw.shape[1]) if args.model == "resnet1d" else InceptionTime(num_classes=num_classes)
    model.to(device)

    pos = train_ds.y.sum(axis=0)
    neg = len(train_ds.y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.5, 10.0), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight if args.task == "super" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_auc, best_epoch, bad = -1.0, -1, 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_auc,val_ap,val_f1\n")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, mv, y in train_loader:
            x, mv, y = x.to(device), mv.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(x, mv), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        yv, pv = evaluate(model, val_loader, device)
        auc, ap, f1 = macro_auc(yv, pv), macro_ap(yv, pv), macro_f1(yv, pv, 0.5)
        line = f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{auc:.6f},{ap:.6f},{f1:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, bad = auc, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_auc": auc}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break
    elapsed = time.time() - t0

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    yv, pv = evaluate(model, val_loader, device)
    thr, val_f1_best = best_threshold(yv, pv)
    yt, pt = evaluate(model, test_loader, device)
    metrics = {
        "best_epoch": best_epoch,
        "val_auc": macro_auc(yv, pv),
        "val_ap": macro_ap(yv, pv),
        "val_f1_best": val_f1_best,
        "threshold": thr,
        "test_auc": macro_auc(yt, pt),
        "test_ap": macro_ap(yt, pt),
        "test_f1_at_0.5": macro_f1(yt, pt, 0.5),
        "test_f1_at_val_thr": macro_f1(yt, pt, thr),
        "test_ece": ece(yt, pt),
        "test_brier": float(np.mean((pt - yt) ** 2)),
        "elapsed_sec": elapsed,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out_dir, "test_pred.npy"), pt)
    np.save(os.path.join(args.out_dir, "test_label.npy"), yt)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

