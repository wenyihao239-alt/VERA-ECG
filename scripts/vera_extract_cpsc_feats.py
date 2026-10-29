#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch

from train_signal import ResNet1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    model = ResNet1d(num_classes=9)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for f in range(5):
        if f == 0:
            d = os.path.join(args.root, "cpsc2018_100"); split = "test"
        else:
            d = os.path.join(args.root, f"cpsc2018_fold{f}_100"); split = "valid"
        x = np.load(os.path.join(d, f"X_{split}.npy"))
        y = np.load(os.path.join(d, f"y_{split}_cpsc.npy"))
        zs = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                xb = x[i:i + 256].astype(np.float32)
                mu = xb.mean(axis=2, keepdims=True); sd = xb.std(axis=2, keepdims=True) + 1e-6
                xb = (xb - mu) / sd
                xb = torch.from_numpy(xb).to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    _, z = model(xb, return_features=True)
                zs.append(z.float().cpu().numpy())
        np.save(os.path.join(args.out_dir, f"cpsc_fold{f}.npy"), np.concatenate(zs, axis=0))
        np.save(os.path.join(args.out_dir, f"cpsc_fold{f}_label.npy"), y)
        print("fold", f, "done", flush=True)


if __name__ == "__main__":
    main()
