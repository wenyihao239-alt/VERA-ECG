#!/usr/bin/env python3
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prepare_image_quality import features_from_array
from prepare_ptbxl import quality_features
from train_image import build_model
from train_signal import ResNet1d, InceptionTime


def corrupt_signal(x, name, level):
    x = x.copy()
    n, leads, T = x.shape
    rng = np.random.default_rng(1234)
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
    return x.astype(np.float32)


def run_signal(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ResNet1d(num_classes=args.num_classes) if args.model == "resnet1d" else InceptionTime(num_classes=args.num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    out_dir = os.path.join(args.out_dir, "signal", args.condition)
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        x_all = np.load(os.path.join(args.data_dir, f"X_{split}.npy"), mmap_mode="r")
        zs, qs = [], []
        for start in range(0, len(x_all), args.chunk):
            x = np.array(x_all[start:start + args.chunk], dtype=np.float32, copy=True)
            if args.condition != "clean":
                mapping = {
                    "noise10": ("noise", 10.0),
                    "baseline03": ("baseline", 0.3),
                    "leaddrop3": ("lead_drop", 3),
                }
                name, level = mapping[args.condition]
                x = corrupt_signal(x, name, level)
            q = np.stack([quality_features(x[i], 100) for i in range(len(x))], axis=0)
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mu) / sd
            with torch.no_grad():
                for i in range(0, len(x), args.batch_size):
                    xb = torch.from_numpy(x[i:i + args.batch_size]).to(device)
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        _, z = model(xb, return_features=True)
                    zs.append(z.float().cpu().numpy())
            qs.append(q)
        z = np.concatenate(zs, axis=0)
        q = np.concatenate(qs, axis=0)
        np.save(os.path.join(out_dir, f"signal_{split}.npy"), z)
        np.save(os.path.join(out_dir, f"q_{split}.npy"), q)
        print(split, z.shape, q.shape, flush=True)


class CorruptImageDataset(Dataset):
    def __init__(self, data_dir, render_dir, split, condition, image_size=224):
        self.meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.render_dir = render_dir
        self.condition = condition
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = os.path.join(self.render_dir, row["split"], f"{int(row['ecg_id'])}.jpg")
        img = Image.open(path).convert("RGB")
        if self.condition == "blur2":
            img = img.filter(ImageFilter.GaussianBlur(2))
        elif self.condition == "crop10":
            w, h = img.size
            dx, dy = int(w * 0.10), int(h * 0.10)
            img = img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.BICUBIC)
        elif self.condition == "jpeg30":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=30)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        qi = features_from_array(np.array(img.convert("L")))
        return self.tf(img), torch.from_numpy(qi)


def run_image(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, args.num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    out_dir = os.path.join(args.out_dir, "image", args.condition)
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        ds = CorruptImageDataset(args.data_dir, args.render_dir, split, args.condition, args.image_size)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        zs, qs = [], []
        with torch.no_grad():
            for x, q in dl:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    z = torch.flatten(backbone(x), 1)
                zs.append(z.float().cpu().numpy())
                qs.append(q.numpy())
        z = np.concatenate(zs, axis=0)
        q = np.concatenate(qs, axis=0)
        np.save(os.path.join(out_dir, f"image_{split}.npy"), z)
        np.save(os.path.join(out_dir, f"qi_{split}.npy"), q)
        print(split, z.shape, q.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["signal", "image"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--render-dir", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.modality == "signal":
        run_signal(args, device)
    else:
        run_image(args, device)
    with open(os.path.join(args.out_dir, f"{args.modality}_{args.condition}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
