#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from train_signal import ece, macro_ap, macro_auc, macro_f1, set_seed
from vera_train_pcbef import load_all

NAMES = ["HR", "PR", "QRS", "QT", "QTc"]


def mlp(in_dim, hidden, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim),
    )


class APEBNet(nn.Module):
    def __init__(self, d_s, d_i, num_classes=5, hidden=256, dropout=0.1):
        super().__init__()
        d = d_s + d_i
        self.diag = mlp(d, hidden, num_classes, dropout)
        self.group = mlp(d, hidden, 4, dropout)
        self.valid = mlp(d, hidden, 5, dropout)
        self.bucket = mlp(d, hidden, 8, dropout)
        self.reg = mlp(d, hidden, 5, dropout)

    def forward(self, zs, zi):
        z = torch.cat([zs, zi], dim=1)
        return {"diag": self.diag(z), "group": self.group(z), "valid": self.valid(z),
                "bucket": self.bucket(z), "reg": self.reg(z)}


class DS(Dataset):
    def __init__(self, sig, img, y, g, ev, mask, bucket, bmask):
        self.zs, self.qs = sig["clean"]
        self.zi, self.qi = img["clean"]
        self.y, self.g, self.ev, self.mask, self.bucket, self.bmask = y, g, ev, mask, bucket, bmask

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.zs[i]), torch.from_numpy(self.zi[i]),
                torch.from_numpy(self.y[i]), torch.from_numpy(self.g[i]),
                torch.from_numpy(self.ev[i]), torch.from_numpy(self.mask[i]),
                torch.from_numpy(self.bucket[i]), torch.from_numpy(self.bmask[i]))


def eval_model(model, dl, device, mu, sd, ev_true):
    model.eval()
    ys, ps, gs, gp, evs, ms, bs, bps, bms, vps = [], [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for zs, zi, y, g, ev, m, b, bm in dl:
            zs, zi = zs.to(device), zi.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(zs, zi)
            ps.append(torch.sigmoid(out["diag"]).float().cpu().numpy())
            gp.append(torch.sigmoid(out["group"]).float().cpu().numpy())
            evs.append(out["reg"].float().cpu().numpy())
            bps.append(out["bucket"].float().cpu().numpy())
            vps.append(torch.sigmoid(out["valid"]).float().cpu().numpy())
            ys.append(y.numpy()); gs.append(g.numpy()); ms.append(m.numpy()); bs.append(b.numpy()); bms.append(bm.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); g = np.concatenate(gs); gp = np.concatenate(gp)
    ev = np.concatenate(evs); m = np.concatenate(ms); b = np.concatenate(bs); bm = np.concatenate(bms)
    vp = np.concatenate(vps)
    out = {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p)}
    out["group_f1"] = macro_f1(g, (gp >= 0.5).astype(int))
    out["group_ap"] = macro_ap(g, gp)
    bp = np.concatenate(bps)
    acc = []
    offsets = [0, 3, 5]; sizes = [3, 2, 3]
    for k in range(3):
        lg = bp[:, offsets[k]:offsets[k] + sizes[k]]
        pred = lg.argmax(axis=1)
        valid = b[:, k] >= 0
        if valid.sum():
            acc.append(float((pred[valid] == b[valid, k]).mean()))
    out["bucket_acc"] = float(np.mean(acc)) if acc else float("nan")
    # evidence MAE / correlation / validity
    ev_pred = ev * sd + mu
    ev_mae, ev_corr = {}, {}
    for j, name in enumerate(["HR", "PR", "QRS", "QT", "QTc"]):
        mm = m[:, j] > 0
        if mm.sum() > 2:
            ev_mae[name] = float(np.mean(np.abs(ev_pred[mm, j] - ev_true[mm, j])))
            ev_corr[name] = float(np.corrcoef(ev_pred[mm, j], ev_true[mm, j])[0, 1])
    out["evidence_mae"] = ev_mae
    out["evidence_corr"] = ev_corr
    vaucs = []
    for j in range(5):
        if m[:, j].sum() > 0 and m[:, j].sum() < len(m):
            vaucs.append(roc_auc_score(m[:, j], vp[:, j]))
    out["validity_auc"] = float(np.mean(vaucs)) if vaucs else float("nan")
    out["validity_acc"] = float(((vp >= 0.5) == (m >= 0.5)).mean())
    # contradiction rate MI/CD/HYP
    support_mi = gp.max(axis=1)
    support_hyp = gp[:, 3]
    support_cd = np.exp(bp[:, 3:5]) / np.exp(bp[:, 3:5]).sum(axis=1, keepdims=True)
    support_cd = support_cd[:, 1]
    contra = ((p[:, 1] > 0.5) & (support_mi < 0.5)) | ((p[:, 3] > 0.5) & (support_cd < 0.5)) | ((p[:, 4] > 0.5) & (support_hyp < 0.5))
    out["contradiction_rate"] = float(contra.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apeb-dir", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--mode", default="a8", choices=["b2", "b3", "a7", "a8", "a8_oneway"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lambda-logic", type=float, default=0.1)
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
    y = {s: np.load(os.path.join(args.data_dir, f"y_{s}_super.npy")) for s in ["train", "val", "test"]}
    g = {s: np.load(os.path.join(args.apeb_dir, f"lead_group_{s}.npy")) for s in ["train", "val", "test"]}
    ev_raw = {s: np.load(os.path.join(args.evidence_dir, f"evidence_{s}.npy")) for s in ["train", "val", "test"]}
    mask = {s: np.load(os.path.join(args.evidence_dir, f"evidence_mask_{s}.npy")) for s in ["train", "val", "test"]}
    bucket = {s: np.load(os.path.join(args.apeb_dir, f"bucket_{s}.npy")) for s in ["train", "val", "test"]}
    bmask = {s: np.load(os.path.join(args.apeb_dir, f"bucket_mask_{s}.npy")) for s in ["train", "val", "test"]}
    mu = np.nanmean(np.where(mask["train"] > 0, ev_raw["train"], np.nan), axis=0)
    sd = np.nanstd(np.where(mask["train"] > 0, ev_raw["train"], np.nan), axis=0) + 1e-6
    ev = {s: np.nan_to_num((ev_raw[s] - mu) / sd, nan=0.0).astype(np.float32) for s in ["train", "val", "test"]}

    ds = {s: DS(sig_tr if s == "train" else (sig_va if s == "val" else sig_te),
               img_tr if s == "train" else (img_va if s == "val" else img_te),
               y[s].astype(np.float32), g[s].astype(np.float32), ev[s], mask[s].astype(np.float32),
               bucket[s], bmask[s].astype(np.float32)) for s in ["train", "val", "test"]}
    dl = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=(s == "train"), num_workers=0) for s in ds}
    num_classes = y["train"].shape[1]
    model = APEBNet(sig_tr["clean"][0].shape[1], img_tr["clean"][0].shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()
    gpos = g["train"].sum(0)
    gneg = len(g["train"]) - gpos
    gw = torch.tensor(np.clip(gneg / np.maximum(gpos, 1), 1.0, 10.0), dtype=torch.float32, device=device)
    gbce = nn.BCEWithLogitsLoss(pos_weight=gw)

    best, best_epoch, bad = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zs, zi, yb, gb, evb, mb, bb, bmb in dl["train"]:
            zs, zi, yb, gb, evb, mb, bb, bmb = [t.to(device) for t in (zs, zi, yb, gb, evb, mb, bb, bmb)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(zs, zi)
                loss = bce(out["diag"], yb)
                if args.mode in ("b2", "b3", "a7", "a8", "a8_oneway"):
                    loss = loss + 0.5 * gbce(out["group"], gb)
                if args.mode in ("a7", "a8", "a8_oneway"):
                    loss = loss + 0.2 * bce(out["valid"], mb)
                    lg = out["bucket"]
                    loss = loss + 0.2 * (F.cross_entropy(lg[:, 0:3], bb[:, 0], ignore_index=-1)
                                         + F.cross_entropy(lg[:, 3:5], bb[:, 1], ignore_index=-1)
                                         + F.cross_entropy(lg[:, 5:8], bb[:, 2], ignore_index=-1))
                    l = F.huber_loss(out["reg"], evb, reduction="none")
                    loss = loss + 0.2 * (l * mb).sum() / (mb.sum() + 1e-6)
                if args.mode in ("b3", "a8", "a8_oneway"):
                    gp = torch.sigmoid(out["group"])
                    p = torch.sigmoid(out["diag"])
                    support_mi = gp.max(dim=1).values
                    support_hyp = gp[:, 3]
                    qrs_logits = out["bucket"][:, 3:5]
                    support_cd = torch.softmax(qrs_logits, dim=1)[:, 1]
                    if args.mode == "a8_oneway":
                        pd = p.detach()
                        l_logic = (F.relu(pd[:, 1] - support_mi - 0.10) ** 2).mean()
                        l_logic = l_logic + (F.relu(pd[:, 3] - support_cd - 0.10) ** 2).mean()
                        l_logic = l_logic + (F.relu(pd[:, 4] - support_hyp - 0.10) ** 2).mean()
                    else:
                        l_logic = (F.relu(p[:, 1] - support_mi).mean() + F.relu(p[:, 3] - support_cd).mean()
                                   + F.relu(p[:, 4] - support_hyp).mean())
                    loss = loss + args.lambda_logic * l_logic
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mva = eval_model(model, dl["val"], device, mu, sd, ev_raw["val"])
        print(f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{mva['auc']:.6f},{mva['group_f1']:.6f}", flush=True)
        if mva["auc"] > best + 1e-4:
            best, best_epoch, bad = mva["auc"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_auc": best}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    mte = eval_model(model, dl["test"], device, mu, sd, ev_raw["test"])
    metrics = {"mode": args.mode, "best_epoch": best_epoch, "val_auc": best, "test": mte,
               "evidence_mu": mu.tolist(), "evidence_sd": sd.tolist(), "args": vars(args)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()






