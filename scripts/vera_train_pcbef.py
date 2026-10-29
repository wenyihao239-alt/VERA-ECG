#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_signal import ece, macro_ap, macro_auc, macro_f1, set_seed

SIG_CONDS = ["clean", "noise10", "baseline03", "leaddrop3"]
IMG_CONDS = ["clean", "blur2", "crop10", "jpeg30"]


def mlp(in_dim, hidden, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim),
    )


class PCBEFNet(nn.Module):
    def __init__(self, d_s, d_i, q_dim, num_classes, mode="pcbef", hidden=256, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.ev_s = mlp(d_s, hidden, 2 * num_classes, dropout)
        self.ev_i = mlp(d_i, hidden, 2 * num_classes, dropout)
        if mode == "pcbef":
            self.gamma_s = mlp(q_dim, hidden // 2, 1, dropout)
            self.gamma_i = mlp(q_dim, hidden // 2, 1, dropout)

    def forward(self, zs, zi, qs, qi, mask_s=None, mask_i=None, gamma_override=None):
        if mask_s is None:
            mask_s = torch.ones(zs.size(0), 1, device=zs.device)
        if mask_i is None:
            mask_i = torch.ones(zi.size(0), 1, device=zi.device)
        k = self.num_classes
        raw_s = self.ev_s(zs)
        raw_i = self.ev_i(zi)
        es_pos = F.softplus(raw_s[:, :k]) + 1e-6
        es_neg = F.softplus(raw_s[:, k:]) + 1e-6
        ei_pos = F.softplus(raw_i[:, :k]) + 1e-6
        ei_neg = F.softplus(raw_i[:, k:]) + 1e-6
        if self.mode == "pcbef":
            if gamma_override is not None:
                gs, gi = gamma_override
            else:
                gs = torch.sigmoid(self.gamma_s(qs))
                gi = torch.sigmoid(self.gamma_i(qi))
            es_pos, es_neg = es_pos * gs, es_neg * gs
            ei_pos, ei_neg = ei_pos * gi, ei_neg * gi
        else:
            gs = torch.ones(zs.size(0), 1, device=zs.device)
            gi = torch.ones(zi.size(0), 1, device=zi.device)
        es_pos, es_neg = es_pos * mask_s, es_neg * mask_s
        ei_pos, ei_neg = ei_pos * mask_i, ei_neg * mask_i
        s_pos, s_neg = es_pos + 1.0, es_neg + 1.0
        i_pos, i_neg = ei_pos + 1.0, ei_neg + 1.0
        bs_pos, bs_neg = es_pos / (s_pos + s_neg), es_neg / (s_pos + s_neg)
        bi_pos, bi_neg = ei_pos / (i_pos + i_neg), ei_neg / (i_pos + i_neg)
        us = 2.0 / (s_pos + s_neg)
        ui = 2.0 / (i_pos + i_neg)
        if self.mode == "pcbef":
            conflict = bs_pos * bi_neg + bs_neg * bi_pos
            denom = (1.0 - conflict).clamp_min(1e-4)
            bp = (bs_pos * bi_pos + bs_pos * ui + us * bi_pos) / denom
            bn = (bs_neg * bi_neg + bs_neg * ui + us * bi_neg) / denom
            u = (us * ui) / denom
            norm = (bp + bn + u).clamp_min(1e-6)
            bp, bn, u = bp / norm, bn / norm, u / norm
            alpha_f = 1.0 + 2.0 * bp / u.clamp_min(1e-4)
            beta_f = 1.0 + 2.0 * bn / u.clamp_min(1e-4)
        else:
            conflict = torch.zeros_like(bs_pos)
            e_pos = es_pos + ei_pos
            e_neg = es_neg + ei_neg
            alpha_f, beta_f = e_pos + 1.0, e_neg + 1.0
            bp = e_pos / (alpha_f + beta_f)
            bn = e_neg / (alpha_f + beta_f)
            u = 2.0 / (alpha_f + beta_f)
        p = (bp + 0.5 * u).clamp(1e-6, 1 - 1e-6)
        return p, {
            "alpha_s": s_pos, "beta_s": s_neg, "alpha_i": i_pos, "beta_i": i_neg,
            "alpha_f": alpha_f, "beta_f": beta_f,
            "gamma_s": gs, "gamma_i": gi, "conflict": conflict, "uncertainty": u,
            "b_pos": bp, "b_neg": bn,
        }


def evidential_loss(alpha, beta, y):
    s = alpha + beta
    return -(y * (torch.log(alpha) - torch.log(s)) + (1 - y) * (torch.log(beta) - torch.log(s))).mean()


def kl_beta(alpha, beta, coef=0.05):
    s = alpha + beta
    kl = (torch.lgamma(s) - torch.lgamma(alpha) - torch.lgamma(beta)
          + (alpha - 1) * torch.digamma(alpha) + (beta - 1) * torch.digamma(beta)
          + (2 - s) * torch.digamma(s))
    return coef * kl.mean()


def load_cond(root, modality, cond, split):
    if modality == "signal":
        z = np.load(os.path.join(root, "signal", cond, f"signal_{split}.npy")).astype(np.float32)
        q = np.load(os.path.join(root, "signal", cond, f"q_{split}.npy")).astype(np.float32)
    else:
        z = np.load(os.path.join(root, "image", cond, f"image_{split}.npy")).astype(np.float32)
        q = np.load(os.path.join(root, "image", cond, f"qi_{split}.npy")).astype(np.float32)
    return z, q


def load_all(root, split, q_dim):
    sig = {c: load_cond(root, "signal", c, split) for c in SIG_CONDS}
    img_raw = {c: load_cond(root, "image", c, split) for c in IMG_CONDS}
    zs_tr = load_cond(root, "signal", "clean", "train")[0]
    zi_tr = load_cond(root, "image", "clean", "train")[0]
    qs_tr = load_cond(root, "signal", "clean", "train")[1]
    qi_tr = load_cond(root, "image", "clean", "train")[1]
    mu_s, sd_s = zs_tr.mean(0, keepdims=True), zs_tr.std(0, keepdims=True) + 1e-6
    mu_i, sd_i = zi_tr.mean(0, keepdims=True), zi_tr.std(0, keepdims=True) + 1e-6
    mu_qs, sd_qs = qs_tr.mean(0, keepdims=True), qs_tr.std(0, keepdims=True) + 1e-6
    qi_tr_p = qi_tr
    if qi_tr_p.shape[1] < q_dim:
        qi_tr_p = np.concatenate([qi_tr_p, np.zeros((len(qi_tr_p), q_dim - qi_tr_p.shape[1]), dtype=np.float32)], axis=1)
    qi_tr_p = qi_tr_p[:, :q_dim]
    mu_qi, sd_qi = qi_tr_p.mean(0, keepdims=True), qi_tr_p.std(0, keepdims=True) + 1e-6
    sig = {c: ((sig[c][0] - mu_s) / sd_s, (sig[c][1] - mu_qs) / sd_qs) for c in SIG_CONDS}
    img = {}
    for c in IMG_CONDS:
        q = img_raw[c][1]
        if q.shape[1] < q_dim:
            q = np.concatenate([q, np.zeros((len(q), q_dim - q.shape[1]), dtype=np.float32)], axis=1)
        q = q[:, :q_dim]
        img[c] = ((img_raw[c][0] - mu_i) / sd_i, (q - mu_qi) / sd_qi)
    return sig, img


class MixedDS(Dataset):
    def __init__(self, sig, img, y, p_clean=0.6, modality_dropout=0.2):
        self.sig, self.img, self.y = sig, img, y.astype(np.float32)
        self.p_clean = p_clean
        self.modality_dropout = modality_dropout

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        cs = "clean" if np.random.rand() < self.p_clean else SIG_CONDS[np.random.randint(1, len(SIG_CONDS))]
        ci = "clean" if np.random.rand() < self.p_clean else IMG_CONDS[np.random.randint(1, len(IMG_CONDS))]
        zs, qs = self.sig[cs]
        zi, qi = self.img[ci]
        corr_s = 0.0 if cs == "clean" else 1.0
        corr_i = 0.0 if ci == "clean" else 1.0
        ms, mi = 1.0, 1.0
        if self.modality_dropout > 0 and np.random.rand() < self.modality_dropout:
            if np.random.rand() < 0.5:
                ms, corr_s = 0.0, 1.0
            else:
                mi, corr_i = 0.0, 1.0
        return (torch.from_numpy(zs[i]), torch.from_numpy(zi[i]), torch.from_numpy(qs[i]), torch.from_numpy(qi[i]),
                torch.from_numpy(self.y[i]), torch.tensor([corr_s], dtype=torch.float32),
                torch.tensor([corr_i], dtype=torch.float32), torch.tensor([ms], dtype=torch.float32),
                torch.tensor([mi], dtype=torch.float32))


def eval_cond(model, sig, img, y, cs, ci, device, q_perm=None, mask_s=1.0, mask_i=1.0):
    zs, qs = sig[cs]
    zi, qi = img[ci]
    if q_perm is not None:
        qs, qi = qs[q_perm], qi[q_perm]
    zs_t = torch.from_numpy(zs).to(device)
    zi_t = torch.from_numpy(zi).to(device)
    qs_t = torch.from_numpy(qs).to(device)
    qi_t = torch.from_numpy(qi).to(device)
    ms = torch.full((len(zs_t), 1), mask_s, device=device)
    mi = torch.full((len(zi_t), 1), mask_i, device=device)
    ps, us = [], []
    with torch.no_grad():
        for i in range(0, len(zs_t), 512):
            p, aux = model(zs_t[i:i + 512], zi_t[i:i + 512], qs_t[i:i + 512], qi_t[i:i + 512], ms[i:i + 512], mi[i:i + 512])
            ps.append(p.float().cpu().numpy())
            us.append(aux["uncertainty"].float().cpu().numpy())
    p = np.concatenate(ps, axis=0)
    u = np.concatenate(us, axis=0)
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p),
            "uncertainty": float(u.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--mode", default="pcbef", choices=["pcbef", "tmc"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--p-clean", type=float, default=0.6)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--lambda-unimodal", type=float, default=0.5)
    ap.add_argument("--lambda-gamma", type=float, default=0.2)
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

    ds = MixedDS(sig_tr, img_tr, y_tr, args.p_clean, args.modality_dropout)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = PCBEFNet(sig_tr["clean"][0].shape[1], img_tr["clean"][0].shape[1], q_dim, y_tr.shape[1], mode=args.mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()

    def robust_val():
        rows = []
        for cs in SIG_CONDS:
            rows.append(eval_cond(model, sig_va, img_va, y_va, cs, "clean", device))
        for ci in IMG_CONDS[1:]:
            rows.append(eval_cond(model, sig_va, img_va, y_va, "clean", ci, device))
        return float(np.mean([r["auc"] for r in rows]))

    best, best_epoch, bad = -1.0, -1, 0
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_robust_auc\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zs, zi, qs, qi, y, corr_s, corr_i, ms, mi in dl:
            zs, zi, qs, qi, y = [t.to(device) for t in (zs, zi, qs, qi, y)]
            corr_s, corr_i, ms, mi = [t.to(device) for t in (corr_s, corr_i, ms, mi)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                p, aux = model(zs, zi, qs, qi, ms, mi)
                loss = evidential_loss(aux["alpha_f"], aux["beta_f"], y)
                loss = loss + args.lambda_unimodal * (
                    evidential_loss(aux["alpha_s"], aux["beta_s"], y)
                    + evidential_loss(aux["alpha_i"], aux["beta_i"], y)
                )
                loss = loss + kl_beta(aux["alpha_f"], aux["beta_f"])
                if args.mode == "pcbef":
                    rel_s, rel_i = 1.0 - corr_s, 1.0 - corr_i
                    loss = loss + args.lambda_gamma * (
                        bce(aux["gamma_s"], rel_s) + bce(aux["gamma_i"], rel_i)
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        robust = robust_val()
        line = f"{epoch},{opt.param_groups[0]['lr']:.6g},{np.mean(losses):.6f},{robust:.6f}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if robust > best + 1e-4:
            best, best_epoch, bad = robust, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_robust_auc": robust}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pairs = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
             ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
             ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30")]
    rows = []
    for cs, ci in pairs:
        r = eval_cond(model, sig_te, img_te, y_te, cs, ci, device)
        rows.append({"sig": cs, "img": ci, **r})
    # missing modality
    miss = []
    for cond, ms, mi in [("signal_only", 1.0, 0.0), ("image_only", 0.0, 1.0)]:
        r = eval_cond(model, sig_te, img_te, y_te, "clean", "clean", device, mask_s=ms, mask_i=mi)
        miss.append({"condition": cond, **r})
    # q permutation
    rng = np.random.default_rng(0)
    qperm = []
    for _ in range(10):
        perm = rng.permutation(len(y_te))
        r = eval_cond(model, sig_te, img_te, y_te, "clean", "clean", device, q_perm=perm)
        qperm.append(r["auc"])
    base_auc = [r for r in rows if r["sig"] == "clean" and r["img"] == "clean"][0]["auc"]
    metrics = {
        "mode": args.mode, "best_epoch": best_epoch, "best_val_robust_auc": best,
        "rows": rows, "missing": miss,
        "q_perm": {"clean_auc": base_auc, "perm_mean": float(np.mean(qperm)), "perm_std": float(np.std(qperm)),
                   "delta": float(base_auc - np.mean(qperm))},
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("rows", "args")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


