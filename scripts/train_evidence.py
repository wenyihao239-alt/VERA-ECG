#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_fusion import FusionNet
from train_fusion_corrupted import load_all
from train_signal import ece, macro_ap, macro_auc, macro_f1, set_seed

NAMES = ["HR", "PR", "QRS", "QT", "QTc"]


class EvidenceDataset(Dataset):
    def __init__(self, sig, img, y, ev, mask):
        self.zs, self.qs = sig["clean"]
        self.zi, self.qi = img["clean"]
        self.y = y.astype(np.float32)
        self.ev = ev.astype(np.float32)
        self.mask = mask.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.zs[i]), torch.from_numpy(self.zi[i]),
            torch.from_numpy(self.qs[i]), torch.from_numpy(self.qi[i]),
            torch.from_numpy(self.y[i]), torch.from_numpy(self.ev[i]), torch.from_numpy(self.mask[i]),
        )


class EvidenceBottleneckWrapper(nn.Module):
    def __init__(self, base, num_classes, num_evidence, hidden=256, dropout=0.1):
        super().__init__()
        self.base = base
        self.diag_head = nn.Sequential(
            nn.Linear(num_evidence, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.residual = nn.Linear(base.d_s + base.d_i, num_classes)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, zs, zi, qs, qi, mask_s=None, mask_i=None):
        _, aux = self.base(zs, zi, qs, qi, mask_s, mask_i)
        ev = aux["evidence"]
        z = aux["z"]
        ev_logits = self.diag_head(ev)
        res_logits = self.residual(z)
        scale = torch.clamp(self.residual_scale, 0.0, 0.5)
        logits = ev_logits + scale * res_logits
        aux.update({"logits_ev": ev_logits, "logits_res": res_logits, "res_scale": scale})
        return logits, aux


def evaluate(model, loader, device, mu, sd, num_evidence, ev_true):
    model.eval()
    ys, ps, evs, ms = [], [], [], []
    with torch.no_grad():
        for zs, zi, qs, qi, y, ev, m in loader:
            zs, zi, qs, qi = [t.to(device) for t in (zs, zi, qs, qi)]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, aux = model(zs, zi, qs, qi)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
            ms.append(m.numpy())
            if num_evidence > 0 and aux.get("evidence") is not None:
                evs.append(aux["evidence"].float().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    mask = np.concatenate(ms, axis=0)
    out = {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p)}
    if evs:
        ev = np.concatenate(evs, axis=0) * sd + mu
        out["evidence"] = {}
        for j, name in enumerate(NAMES):
            m = mask[:, j] > 0
            if m.sum() == 0:
                continue
            err = ev[m, j] - ev_true[m, j]
            out["evidence"][name] = {
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "corr": float(np.corrcoef(ev[m, j], ev_true[m, j])[0, 1]) if m.sum() > 2 else float("nan"),
                "coverage": float(m.mean()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--mode", default="quality", choices=["quality", "gated_md"])
    ap.add_argument("--evidence-mode", default="reg", choices=["none", "reg", "bottleneck", "evidence_bottleneck"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lambda-evidence", type=float, default=0.2)
    ap.add_argument("--lambda-residual", type=float, default=0.01)
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
    ev_tr = np.load(os.path.join(args.evidence_dir, "evidence_train.npy"))
    ev_va = np.load(os.path.join(args.evidence_dir, "evidence_val.npy"))
    ev_te = np.load(os.path.join(args.evidence_dir, "evidence_test.npy"))
    m_tr = np.load(os.path.join(args.evidence_dir, "evidence_mask_train.npy"))
    m_va = np.load(os.path.join(args.evidence_dir, "evidence_mask_val.npy"))
    m_te = np.load(os.path.join(args.evidence_dir, "evidence_mask_test.npy"))
    mu = np.nanmean(np.where(m_tr > 0, ev_tr, np.nan), axis=0)
    sd = np.nanstd(np.where(m_tr > 0, ev_tr, np.nan), axis=0) + 1e-6
    ev_tr = np.nan_to_num((ev_tr - mu) / sd, nan=0.0)
    ev_va = np.nan_to_num((ev_va - mu) / sd, nan=0.0)
    ev_te_n = np.nan_to_num((ev_te - mu) / sd, nan=0.0)

    ds_tr = EvidenceDataset(sig_tr, img_tr, y_tr, ev_tr, m_tr)
    ds_va = EvidenceDataset(sig_va, img_va, y_va, ev_va, m_va)
    ds_te = EvidenceDataset(sig_te, img_te, y_te, ev_te_n, m_te)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_ev = 0 if args.evidence_mode == "none" else 5
    base = FusionNet(args.mode, sig_tr["clean"][0].shape[1], img_tr["clean"][0].shape[1], q_dim, y_tr.shape[1],
                     num_evidence=num_ev, bottleneck=(args.evidence_mode == "bottleneck"))
    if args.evidence_mode == "evidence_bottleneck":
        model = EvidenceBottleneckWrapper(base, y_tr.shape[1], num_ev).to(device)
    else:
        model = base.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(reduction="none")

    best = -1.0
    best_epoch = -1
    bad = 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_auc\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zs, zi, qs, qi, y, ev, m in dl_tr:
            zs, zi, qs, qi, y, ev, m = [t.to(device) for t in (zs, zi, qs, qi, y, ev, m)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, aux = model(zs, zi, qs, qi)
                loss = bce(logits, y)
                if num_ev > 0 and aux.get("evidence") is not None:
                    l = huber(aux["evidence"], ev)
                    loss = loss + args.lambda_evidence * (l * m).sum() / (m.sum() + 1e-6)
                if args.evidence_mode == "evidence_bottleneck":
                    loss = loss + args.lambda_residual * (model.residual_scale ** 2 + model.residual.weight.pow(2).sum())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mva = evaluate(model, dl_va, device, mu, sd, num_ev, ev_va)
        line = f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{mva['auc']:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if mva["auc"] > best + 1e-4:
            best = mva["auc"]
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_auc": best}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    mte = evaluate(model, dl_te, device, mu, sd, num_ev, ev_te)
    metrics = {"best_epoch": best_epoch, "val_auc": best, "test": mte, "args": vars(args)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



