#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_signal import best_threshold, ece, macro_ap, macro_auc, macro_f1, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ECGImagePreDataset(Dataset):
    def __init__(self, data_dir, render_dir, split, task, train=False, height=384, width=512):
        self.meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.labels = np.load(os.path.join(data_dir, f"y_{split}_{task}.npy")).astype(np.float32)
        self.render_dir = render_dir
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop((height, width), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = os.path.join(self.render_dir, row["split"], f"{int(row['ecg_id'])}.jpg")
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.from_numpy(self.labels[i])


def build_model(name, num_classes, pretrained=True):
    if name == "resnet34":
        w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(name)
    return m


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--model", default="resnet34", choices=["resnet34", "resnet50", "convnext_tiny"])
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = ECGImagePreDataset(args.data_dir, args.render_dir, "train", args.task, True, args.height, args.width)
    val_ds = ECGImagePreDataset(args.data_dir, args.render_dir, "val", args.task, False, args.height, args.width)
    test_ds = ECGImagePreDataset(args.data_dir, args.render_dir, "test", args.task, False, args.height, args.width)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_ds.labels.shape[1]
    model = build_model(args.model, num_classes, not args.no_pretrained).to(device)
    pos = train_ds.labels.sum(axis=0)
    neg = len(train_ds.labels) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.5, 10.0), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight if args.task == "super" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_auc, best_epoch, bad = -1.0, -1, 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_auc,val_ap,val_f1\n")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
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
