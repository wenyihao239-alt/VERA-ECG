#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch

from train_signal_mv import ResNet1dMV
from vera_prepare_mv import stats_for_split
from vera_mv_utils import stats_hp


def extract(model, x, mv_mode, mv_mean, mv_std, hp_mean, hp_std, device, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            raw = x[i:i + batch].astype(np.float32)
            mu = raw.mean(axis=2, keepdims=True)
            sd = raw.std(axis=2, keepdims=True) + 1e-6
            xb = (raw - mu) / sd
            if mv_mode == "zero":
                mb = np.zeros((len(xb), 36), dtype=np.float32)
            elif mv_mode == "raw":
                mb = (stats_for_split(raw) - mv_mean) / mv_std
            else:
                h = stats_hp(raw)
                mb = (h - hp_mean) / hp_std
            xb = torch.from_numpy(xb).to(device)
            mb = torch.from_numpy(mb.astype(np.float32)).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _, z = model(xb, mb, return_features=True)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ptb-dir", required=True)
    ap.add_argument("--cpsc-root", required=True)
    ap.add_argument("--hp-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mv-mode", required=True, choices=["zero", "hp_std", "raw"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    model = ResNet1dMV(num_classes=9, mv_dim=36)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    mv_mean = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_mean.npy")).reshape(-1)
    mv_std = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_std.npy")).reshape(-1)
    hp_mean = hp_std = None
    if args.mv_mode in ("hp_std", "raw"):
        d = np.load(args.hp_cache)
        hp_mean, hp_std = d["mean"], d["std"]

    for split in ["train", "val"]:
        x = np.load(os.path.join(args.ptb_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(args.ptb_dir, f"y_{split}_cpsc.npy"))
        z = extract(model, x, args.mv_mode, mv_mean, mv_std, hp_mean, hp_std, device)
        np.save(os.path.join(args.out_dir, f"signal_{split}.npy"), z)
        np.save(os.path.join(args.out_dir, f"y_{split}_cpsc.npy"), y)
        print("ptb", split, z.shape, flush=True)

    for f in range(5):
        if f == 0:
            d = os.path.join(args.cpsc_root, "cpsc2018_100")
            split = "test"
        else:
            d = os.path.join(args.cpsc_root, f"cpsc2018_fold{f}_100")
            split = "valid"
        x = np.load(os.path.join(d, f"X_{split}.npy"))
        y = np.load(os.path.join(d, f"y_{split}_cpsc.npy"))
        z = extract(model, x, args.mv_mode, mv_mean, mv_std, hp_mean, hp_std, device)
        np.save(os.path.join(args.out_dir, f"cpsc_fold{f}.npy"), z)
        np.save(os.path.join(args.out_dir, f"cpsc_fold{f}_label.npy"), y)
        print("cpsc", f, z.shape, flush=True)


if __name__ == "__main__":
    main()
