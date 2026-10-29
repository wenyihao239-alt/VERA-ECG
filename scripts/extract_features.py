#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_image import ECGImageDataset, build_model
from train_signal import ECGArray, InceptionTime, ResNet1d


def signal_features(args, device):
    if args.model == "resnet1d":
        model = ResNet1d(num_classes=args.num_classes)
    else:
        model = InceptionTime(num_classes=args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    out = {}
    for split in ["train", "val", "test"]:
        ds = ECGArray(args.data_dir, split, args.task, "none", train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        zs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    _, z = model(x, return_features=True)
                zs.append(z.float().cpu().numpy())
        z = np.concatenate(zs, axis=0)
        out[split] = z
        np.save(os.path.join(args.out_dir, f"signal_{split}.npy"), z)
        print(split, z.shape, flush=True)
    return out


def image_features(args, device):
    model = build_model(args.model, args.num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    out = {}
    for split in ["train", "val", "test"]:
        ds = ECGImageDataset(args.data_dir, args.render_dir, split, args.task, train=False, image_size=args.image_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        zs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    z = torch.flatten(backbone(x), 1)
                zs.append(z.float().cpu().numpy())
        z = np.concatenate(zs, axis=0)
        out[split] = z
        np.save(os.path.join(args.out_dir, f"image_{split}.npy"), z)
        print(split, z.shape, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["signal", "image"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--render-dir", default="")
    ap.add_argument("--task", default="super")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--model", default="resnet1d")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.modality == "signal":
        signal_features(args, device)
    else:
        image_features(args, device)
    with open(os.path.join(args.out_dir, f"{args.modality}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
