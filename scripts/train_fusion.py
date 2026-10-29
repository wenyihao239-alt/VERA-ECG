#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_signal import best_threshold, ece, macro_ap, macro_auc, macro_f1, set_seed


class FeatureDataset(Dataset):
    def __init__(self, zs, zi, qs, qi, y):
        self.zs = torch.from_numpy(zs.astype(np.float32))
        self.zi = torch.from_numpy(zi.astype(np.float32))
        self.qs = torch.from_numpy(qs.astype(np.float32))
        self.qi = torch.from_numpy(qi.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.zs[i], self.zi[i], self.qs[i], self.qi[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, mode, d_s, d_i, q_dim, num_classes, hidden=256, dropout=0.1, num_evidence=0, bottleneck=False):
        super().__init__()
        self.mode = mode
        self.d_s = d_s
        self.d_i = d_i
        self.q_dim = q_dim
        self.num_classes = num_classes
        self.num_evidence = num_evidence
        self.bottleneck = bottleneck
        head_in = d_s + d_i + (num_evidence if bottleneck else 0)
        self.evidence_head = MLP(d_s + d_i, hidden, num_evidence, dropout) if num_evidence > 0 else None
        if mode == "early":
            self.head = MLP(head_in, hidden, num_classes, dropout)
        elif mode == "late":
            self.head_s = MLP(d_s, hidden, num_classes, dropout)
            self.head_i = MLP(d_i, hidden, num_classes, dropout)
        elif mode in ("gated", "gated_md"):
            self.gate = MLP(d_s + d_i + 2, hidden, 1, dropout)
            self.head = MLP(head_in, hidden, num_classes, dropout)
        elif mode == "quality":
            self.probe_s = MLP(d_s, hidden, num_classes, dropout)
            self.probe_i = MLP(d_i, hidden, num_classes, dropout)
            self.q_head_s = MLP(q_dim, hidden // 2, 1, dropout)
            self.q_head_i = MLP(q_dim, hidden // 2, 1, dropout)
            extra = 2 * q_dim + 5
            self.gate = MLP(d_s + d_i + extra, hidden, 1, dropout)
            self.head = MLP(head_in, hidden, num_classes, dropout)
        else:
            raise ValueError(mode)

    def _head(self, z):
        ev = self.evidence_head(z) if self.evidence_head is not None else None
        zz = torch.cat([z, ev], dim=1) if (ev is not None and self.bottleneck) else z
        return self.head(zz), ev

    def forward(self, zs, zi, qs, qi, mask_s=None, mask_i=None):
        if mask_s is None:
            mask_s = torch.ones(zs.size(0), 1, device=zs.device)
        if mask_i is None:
            mask_i = torch.ones(zi.size(0), 1, device=zi.device)
        zs = zs * mask_s
        zi = zi * mask_i
        if self.mode == "early":
            z = torch.cat([zs, zi], dim=1)
            logits, ev = self._head(z)
            return logits, {"evidence": ev, "z": z}
        if self.mode == "late":
            ls = self.head_s(zs)
            li = self.head_i(zi)
            logits = (ls * mask_s + li * mask_i) / (mask_s + mask_i + 1e-6)
            z = torch.cat([zs, zi], dim=1)
            ev = self.evidence_head(z) if self.evidence_head is not None else None
            return logits, {"logits_s": ls, "logits_i": li, "evidence": ev, "z": z}
        if self.mode in ("gated", "gated_md"):
            g = torch.sigmoid(self.gate(torch.cat([zs, zi, mask_s, mask_i], dim=1)))
            g = torch.where(mask_s < 0.5, torch.zeros_like(g), g)
            g = torch.where(mask_i < 0.5, torch.ones_like(g), g)
            if getattr(self, "force_gate", None) is not None:
                g = torch.full_like(g, float(self.force_gate))
            z = torch.cat([g * zs, (1 - g) * zi], dim=1)
            logits, ev = self._head(z)
            return logits, {"gate": g, "evidence": ev, "z": z}
        qp_s_logit = self.q_head_s(qs)
        qp_i_logit = self.q_head_i(qi)
        qp_s = torch.sigmoid(qp_s_logit)
        qp_i = torch.sigmoid(qp_i_logit)
        d = min(zs.size(1), zi.size(1)); cos = F.cosine_similarity(zs[:, :d], zi[:, :d], dim=1, eps=1e-6).unsqueeze(1)
        ls = self.probe_s(zs)
        li = self.probe_i(zi)
        ps = torch.sigmoid(ls)
        pi = torch.sigmoid(li)
        agree = (1.0 - (ps - pi).abs().mean(dim=1, keepdim=True))
        ent = -(ps.clamp(1e-6, 1 - 1e-6) * ps.clamp(1e-6, 1 - 1e-6).log() + (1 - ps).clamp(1e-6, 1 - 1e-6) * (1 - ps).clamp(1e-6, 1 - 1e-6).log())
        ent = ent.mean(dim=1, keepdim=True)
        gate_in = torch.cat([zs, zi, qs, qi, cos, agree, ent, qp_s, qp_i], dim=1)
        g_logit = self.gate(gate_in)
        g = torch.sigmoid(g_logit)
        g = torch.where(mask_s < 0.5, torch.zeros_like(g), g)
        g = torch.where(mask_i < 0.5, torch.ones_like(g), g)
        if getattr(self, "force_gate", None) is not None:
            g = torch.full_like(g, float(self.force_gate))
        z = torch.cat([g * zs, (1 - g) * zi], dim=1)
        logits, ev = self._head(z)
        return logits, {"gate": g, "gate_logit": g_logit, "q_logit_s": qp_s_logit, "q_logit_i": qp_i_logit, "q_pred_s": qp_s, "q_pred_i": qp_i, "logits_s": ls, "logits_i": li, "evidence": ev, "z": z}



def load_features(args):
    zs = {s: np.load(os.path.join(args.signal_features, f"signal_{s}.npy")) for s in ["train", "val", "test"]}
    zi = {s: np.load(os.path.join(args.image_features, f"image_{s}.npy")) for s in ["train", "val", "test"]}
    qs = {s: np.load(os.path.join(args.data_dir, f"q_{s}.npy")) for s in ["train", "val", "test"]}
    y = {s: np.load(os.path.join(args.data_dir, f"y_{s}_{args.task}.npy")) for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        n = len(y[s])
        if len(zs[s]) != n or len(zi[s]) != n or len(qs[s]) != n:
            raise ValueError(f"length mismatch in {s}: y={n}, zs={len(zs[s])}, zi={len(zi[s])}, qs={len(qs[s])}")
    if args.image_quality:
        qi = {s: np.load(os.path.join(args.image_quality, f"qi_{s}.npy")) for s in ["train", "val", "test"]}
    else:
        qi = {s: np.zeros_like(qs[s]) for s in ["train", "val", "test"]}
    q_dim = qs["train"].shape[1]
    for s in ["train", "val", "test"]:
        if qi[s].shape[1] < q_dim:
            pad = np.zeros((len(qi[s]), q_dim - qi[s].shape[1]), dtype=np.float32)
            qi[s] = np.concatenate([qi[s], pad], axis=1)
        elif qi[s].shape[1] > q_dim:
            qi[s] = qi[s][:, :q_dim]
    return zs, zi, qs, qi, y


def normalize(train, val, test):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sd, (val - mu) / sd, (test - mu) / sd


def evaluate(model, loader, device):
    model.eval()
    ys, ps, gates = [], [], []
    with torch.no_grad():
        for zs, zi, qs, qi, y in loader:
            zs, zi, qs, qi = [t.to(device, non_blocking=True) for t in (zs, zi, qs, qi)]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, aux = model(zs, zi, qs, qi)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
            if "gate" in aux:
                gates.append(aux["gate"].float().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    g = np.concatenate(gates, axis=0) if gates else None
    return y, p, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--signal-features", required=True)
    ap.add_argument("--image-features", required=True)
    ap.add_argument("--image-quality", default="")
    ap.add_argument("--task", default="super")
    ap.add_argument("--mode", default="quality", choices=["early", "late", "gated", "gated_md", "quality"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--lambda-gate", type=float, default=0.1)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    zs, zi, qs, qi, y = load_features(args)
    zs["train"], zs["val"], zs["test"] = normalize(zs["train"], zs["val"], zs["test"])
    zi["train"], zi["val"], zi["test"] = normalize(zi["train"], zi["val"], zi["test"])
    qs["train"], qs["val"], qs["test"] = normalize(qs["train"], qs["val"], qs["test"])
    if args.image_quality:
        qi["train"], qi["val"], qi["test"] = normalize(qi["train"], qi["val"], qi["test"])

    ds = {s: FeatureDataset(zs[s], zi[s], qs[s], qi[s], y[s]) for s in ["train", "val", "test"]}
    dl = {
        "train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
        "val": DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
        "test": DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
    }

    num_classes = y["train"].shape[1]
    model = FusionNet(args.mode, zs["train"].shape[1], zi["train"].shape[1], qs["train"].shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_epoch = -1
    bad = 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_auc,val_ap,val_f1\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zs_b, zi_b, qs_b, qi_b, yb in dl["train"]:
            zs_b, zi_b, qs_b, qi_b, yb = [t.to(device, non_blocking=True) for t in (zs_b, zi_b, qs_b, qi_b, yb)]
            mask_s = torch.ones(zs_b.size(0), 1, device=device)
            mask_i = torch.ones(zi_b.size(0), 1, device=device)
            if args.mode in ("gated_md", "quality") and args.modality_dropout > 0:
                drop = torch.rand(zs_b.size(0), 1, device=device) < args.modality_dropout
                which = torch.randint(0, 2, (zs_b.size(0), 1), device=device)
                mask_s = torch.where(drop & (which == 0), torch.zeros_like(mask_s), mask_s)
                mask_i = torch.where(drop & (which == 1), torch.zeros_like(mask_i), mask_i)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, aux = model(zs_b, zi_b, qs_b, qi_b, mask_s, mask_i)
                loss = loss_fn(logits, yb)
                if args.mode == "quality" and args.lambda_probe > 0:
                    loss = loss + args.lambda_probe * (
                        loss_fn(aux["logits_s"], yb) + loss_fn(aux["logits_i"], yb)
                    )
                if args.mode == "quality" and args.lambda_gate > 0:
                    with torch.no_grad():
                        ps = torch.sigmoid(aux["logits_s"])
                        pi = torch.sigmoid(aux["logits_i"])
                        cs = ((ps > 0.5) == (yb > 0.5)).float().mean(dim=1, keepdim=True)
                        ci = ((pi > 0.5) == (yb > 0.5)).float().mean(dim=1, keepdim=True)
                        target = cs / (cs + ci + 1e-6)
                    loss = loss + args.lambda_gate * (
                        F.binary_cross_entropy_with_logits(aux["q_logit_s"], cs)
                        + F.binary_cross_entropy_with_logits(aux["q_logit_i"], ci)
                        + F.binary_cross_entropy_with_logits(aux["gate_logit"], target)
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        yv, pv, _ = evaluate(model, dl["val"], device)
        auc = macro_auc(yv, pv)
        ap = macro_ap(yv, pv)
        f1 = macro_f1(yv, pv, 0.5)
        line = f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{auc:.6f},{ap:.6f},{f1:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_auc": auc}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    yv, pv, gv = evaluate(model, dl["val"], device)
    thr, val_f1_best = best_threshold(yv, pv)
    yt, pt, gt = evaluate(model, dl["test"], device)
    metrics = {
        "best_epoch": best_epoch,
        "mode": args.mode,
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
        "n_train": len(ds["train"]),
        "n_val": len(ds["val"]),
        "n_test": len(ds["test"]),
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out_dir, "test_pred.npy"), pt)
    np.save(os.path.join(args.out_dir, "test_label.npy"), yt)
    np.save(os.path.join(args.out_dir, "val_pred.npy"), pv)
    np.save(os.path.join(args.out_dir, "val_label.npy"), yv)
    if gt is not None:
        np.save(os.path.join(args.out_dir, "test_gate.npy"), gt)
    if gv is not None:
        np.save(os.path.join(args.out_dir, "val_gate.npy"), gv)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()










