#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_signal import InceptionTime, ResNet1d, ece, macro_ap, macro_auc, macro_f1


class TestArray(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].copy()
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        return torch.from_numpy((x - mu) / sd), torch.from_numpy(self.y[i])


def corrupt(x, name, level):
    x = x.copy()
    n, leads, T = x.shape
    rng = np.random.default_rng(12345)
    if name == "noise":
        sig = np.mean(x ** 2, axis=(1, 2), keepdims=True) + 1e-8
        noise = sig / (10 ** (level / 10.0))
        x = x + rng.normal(0, np.sqrt(noise), size=x.shape).astype(np.float32)
    elif name == "baseline":
        t = np.arange(T, dtype=np.float32) / 100.0
        f = rng.uniform(0.05, 0.5, size=(n, 1, 1)).astype(np.float32)
        x = x + (level * np.sin(2 * np.pi * f * t[None, None, :])).astype(np.float32)
    elif name == "lead_drop":
        for i in range(n):
            idx = rng.choice(leads, size=min(level, leads), replace=False)
            x[i, idx] = 0
    elif name == "scale":
        x = x * level
    elif name == "shift":
        x = np.roll(x, int(level), axis=2)
    return x


def evaluate(model, x, y, device, batch_size=256):
    loader = DataLoader(TestArray(x, y), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xb)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(yb.numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--model", default="resnet1d", choices=["resnet1d", "inceptiontime"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    num_classes = y.shape[1]
    model = ResNet1d(num_classes=num_classes) if args.model == "resnet1d" else InceptionTime(num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    rows = [{"corruption": "clean", "level": 0, **evaluate(model, x, y, device)}]
    clean_auc = rows[0]["auc"]
    grid = [
        ("noise", [20, 10, 5]),
        ("baseline", [0.1, 0.3, 0.5]),
        ("lead_drop", [1, 3, 6]),
        ("scale", [0.5, 1.5]),
        ("shift", [50, 200]),
    ]
    for name, levels in grid:
        for level in levels:
            xc = corrupt(x, name, level)
            m = evaluate(model, xc, y, device)
            rows.append({"corruption": name, "level": level, **m, "delta_auc": clean_auc - m["auc"]})
            print(name, level, m, flush=True)
    with open(os.path.join(args.out_dir, "robustness.csv"), "w", encoding="utf-8") as f:
        keys = ["corruption", "level", "auc", "ap", "f1", "ece", "delta_auc"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"clean": rows[0], "rows": rows}, f, ensure_ascii=False, indent=2)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
