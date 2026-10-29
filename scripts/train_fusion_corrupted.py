#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_fusion import FusionNet
from train_signal import ece, macro_ap, macro_auc, macro_f1, set_seed

SIG_CONDS = ["clean", "noise10", "baseline03", "leaddrop3"]
IMG_CONDS = ["clean", "blur2", "crop10", "jpeg30"]


def load_cond(root, modality, cond, split):
    if modality == "signal":
        z = np.load(os.path.join(root, "signal", cond, f"signal_{split}.npy"))
        q = np.load(os.path.join(root, "signal", cond, f"q_{split}.npy"))
    else:
        z = np.load(os.path.join(root, "image", cond, f"image_{split}.npy"))
        q = np.load(os.path.join(root, "image", cond, f"qi_{split}.npy"))
    return z.astype(np.float32), q.astype(np.float32)


def normalize_with(train, arrs):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return [(a - mu) / sd for a in arrs]


def load_all(root, split, q_dim):
    sig, img = {}, {}
    for c in SIG_CONDS:
        sig[c] = load_cond(root, "signal", c, split)
    for c in IMG_CONDS:
        img[c] = load_cond(root, "image", c, split)
    sig_train = load_cond(root, "signal", "clean", "train")
    img_train = load_cond(root, "image", "clean", "train")
    sig_zs = normalize_with(sig_train[0], [sig[c][0] for c in SIG_CONDS])
    sig_qs = normalize_with(sig_train[1], [sig[c][1] for c in SIG_CONDS])
    img_zs = normalize_with(img_train[0], [img[c][0] for c in IMG_CONDS])
    img_qs = []
    for c in IMG_CONDS:
        q = img[c][1]
        if q.shape[1] < q_dim:
            q = np.concatenate([q, np.zeros((len(q), q_dim - q.shape[1]), dtype=np.float32)], axis=1)
        img_qs.append(q[:, :q_dim])
    img_qs = normalize_with(img_train[1][:, :q_dim] if img_train[1].shape[1] >= q_dim else np.concatenate(
        [img_train[1], np.zeros((len(img_train[1]), q_dim - img_train[1].shape[1]), dtype=np.float32)], axis=1
    ), img_qs)
    sig = {c: (sig_zs[i], sig_qs[i]) for i, c in enumerate(SIG_CONDS)}
    img = {c: (img_zs[i], img_qs[i]) for i, c in enumerate(IMG_CONDS)}
    return sig, img


class MultiCondDataset(Dataset):
    def __init__(self, sig, img, y, train=True, p_clean=0.6, modality_dropout=0.2):
        self.sig = sig
        self.img = img
        self.y = y.astype(np.float32)
        self.train = train
        self.p_clean = p_clean
        self.modality_dropout = modality_dropout
        self.sig_conds = SIG_CONDS
        self.img_conds = IMG_CONDS

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.train:
            cs = "clean" if np.random.rand() < self.p_clean else self.sig_conds[np.random.randint(1, len(self.sig_conds))]
            ci = "clean" if np.random.rand() < self.p_clean else self.img_conds[np.random.randint(1, len(self.img_conds))]
        else:
            cs, ci = "clean", "clean"
        zs, qs = self.sig[cs]
        zi, qi = self.img[ci]
        corr_s = 0.0 if cs == "clean" else 1.0
        corr_i = 0.0 if ci == "clean" else 1.0
        mask_s = 1.0
        mask_i = 1.0
        if self.train and self.modality_dropout > 0 and np.random.rand() < self.modality_dropout:
            if np.random.rand() < 0.5:
                mask_s = 0.0
                corr_s = 1.0
            else:
                mask_i = 0.0
                corr_i = 1.0
        return (
            torch.from_numpy(zs[i]),
            torch.from_numpy(zi[i]),
            torch.from_numpy(qs[i]),
            torch.from_numpy(qi[i]),
            torch.from_numpy(self.y[i]),
            torch.tensor([corr_s], dtype=torch.float32),
            torch.tensor([corr_i], dtype=torch.float32),
            torch.tensor([mask_s], dtype=torch.float32),
            torch.tensor([mask_i], dtype=torch.float32),
        )


def eval_cond(model, sig, img, y, cs, ci, device, batch_size=512):
    zs = torch.from_numpy(sig[cs][0]).to(device)
    qs = torch.from_numpy(sig[cs][1]).to(device)
    zi = torch.from_numpy(img[ci][0]).to(device)
    qi = torch.from_numpy(img[ci][1]).to(device)
    ps = []
    with torch.no_grad():
        for i in range(0, len(zs), batch_size):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, _ = model(zs[i:i + batch_size], zi[i:i + batch_size], qs[i:i + batch_size], qi[i:i + batch_size])
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
    p = np.concatenate(ps, axis=0)
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p)}


def robust_eval(model, sig, img, y, device):
    pairs = [("clean", "clean")]
    pairs += [(c, "clean") for c in SIG_CONDS[1:]]
    pairs += [("clean", c) for c in IMG_CONDS[1:]]
    rows = []
    for cs, ci in pairs:
        m = eval_cond(model, sig, img, y, cs, ci, device)
        rows.append({"sig": cs, "img": ci, **m})
    return rows, float(np.mean([r["auc"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--mode", required=True, choices=["gated_md", "quality"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--p-clean", type=float, default=0.6)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--lambda-q", type=float, default=0.5)
    ap.add_argument("--lambda-gate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig_tr, img_tr = load_all(args.feature_root, "train", q_dim)
    sig_va, img_va = load_all(args.feature_root, "val", q_dim)
    sig_te, img_te = load_all(args.feature_root, "test", q_dim)
    y_tr = np.load(os.path.join(args.data_dir, f"y_train_{args.task}.npy"))
    y_va = np.load(os.path.join(args.data_dir, f"y_val_{args.task}.npy"))
    y_te = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))

    ds = MultiCondDataset(sig_tr, img_tr, y_tr, train=True, p_clean=args.p_clean, modality_dropout=args.modality_dropout)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    num_classes = y_tr.shape[1]
    model = FusionNet(args.mode, sig_tr["clean"][0].shape[1], img_tr["clean"][0].shape[1], q_dim, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()

    best = -1.0
    best_epoch = -1
    bad = 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_robust_auc\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zs, zi, qs, qi, yb, corr_s, corr_i, mask_s, mask_i in dl:
            zs, zi, qs, qi, yb = [t.to(device) for t in (zs, zi, qs, qi, yb)]
            corr_s, corr_i, mask_s, mask_i = [t.to(device) for t in (corr_s, corr_i, mask_s, mask_i)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, aux = model(zs, zi, qs, qi, mask_s, mask_i)
                loss = bce(logits, yb)
                if args.mode == "quality":
                    bs = F.binary_cross_entropy_with_logits(aux["logits_s"], yb, reduction="none").mean(dim=1, keepdim=True)
                    bi = F.binary_cross_entropy_with_logits(aux["logits_i"], yb, reduction="none").mean(dim=1, keepdim=True)
                    loss = loss + args.lambda_probe * (
                        (bs * mask_s).sum() / (mask_s.sum() + 1e-6) + (bi * mask_i).sum() / (mask_i.sum() + 1e-6)
                    )
                    rel_s = 1.0 - corr_s
                    rel_i = 1.0 - corr_i
                    loss = loss + args.lambda_q * (bce(aux["q_logit_s"], rel_s) + bce(aux["q_logit_i"], rel_i))
                    target = rel_s / (rel_s + rel_i + 1e-6)
                    valid = mask_s * mask_i
                    loss = loss + args.lambda_gate * (F.binary_cross_entropy_with_logits(
                        aux["gate_logit"], target, reduction="none") * valid).sum() / (valid.sum() + 1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        rows, robust = robust_eval(model, sig_va, img_va, y_va, device)
        line = f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{robust:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if robust > best + 1e-4:
            best = robust
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_robust_auc": robust}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    rows_te, robust_te = robust_eval(model, sig_te, img_te, y_te, device)
    clean = [r for r in rows_te if r["sig"] == "clean" and r["img"] == "clean"][0]
    sig_bad = [r for r in rows_te if r["sig"] != "clean" and r["img"] == "clean"]
    img_bad = [r for r in rows_te if r["sig"] == "clean" and r["img"] != "clean"]
    metrics = {
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_robust_auc": best,
        "clean": clean,
        "signal_corrupted_mean_auc": float(np.mean([r["auc"] for r in sig_bad])),
        "image_corrupted_mean_auc": float(np.mean([r["auc"] for r in img_bad])),
        "robust_mean_auc": robust_te,
        "rows": rows_te,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
