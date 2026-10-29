#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_signal import ece, macro_ap, macro_auc, macro_f1, set_seed
from vera_train_pcbef import MixedDS, SIG_CONDS, IMG_CONDS, load_all


def mlp(in_dim, hidden, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim),
    )


def evid_elem(alpha, beta, y):
    s = alpha + beta
    return (-(y * (torch.log(alpha) - torch.log(s)) + (1 - y) * (torch.log(beta) - torch.log(s)))).mean(dim=1)


def kl_beta(alpha, beta, coef=0.05):
    s = alpha + beta
    kl = (torch.lgamma(s) - torch.lgamma(alpha) - torch.lgamma(beta)
          + (alpha - 1) * torch.digamma(alpha) + (beta - 1) * torch.digamma(beta)
          + (2 - s) * torch.digamma(s))
    return coef * kl.mean()


class PCBEFv2Net(nn.Module):
    def __init__(self, d_s, d_i, q_dim, num_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ev_s = mlp(d_s, hidden, 2 * num_classes, dropout)
        self.ev_i = mlp(d_i, hidden, 2 * num_classes, dropout)
        self.gamma_s = mlp(q_dim, hidden // 2, 1, dropout)
        self.gamma_i = mlp(q_dim, hidden // 2, 1, dropout)

    def forward(self, zs, zi, qs, qi, mask_s=None, mask_i=None):
        if mask_s is None:
            mask_s = torch.ones(zs.size(0), 1, device=zs.device)
        if mask_i is None:
            mask_i = torch.ones(zi.size(0), 1, device=zi.device)
        k = self.num_classes
        rs, ri = self.ev_s(zs), self.ev_i(zi)
        es_pos = F.softplus(rs[:, :k]) + 1e-6
        es_neg = F.softplus(rs[:, k:]) + 1e-6
        ei_pos = F.softplus(ri[:, :k]) + 1e-6
        ei_neg = F.softplus(ri[:, k:]) + 1e-6
        gs_logit = self.gamma_s(qs) + 2.0
        gi_logit = self.gamma_i(qi) + 2.0
        gs = 0.70 + 0.30 * torch.sigmoid(gs_logit)
        gi = 0.45 + 0.55 * torch.sigmoid(gi_logit)
        es_pos_m = es_pos * mask_s
        es_neg_m = es_neg * mask_s
        ei_pos_m = ei_pos * mask_i
        ei_neg_m = ei_neg * mask_i
        e_pos_tmc = es_pos_m + ei_pos_m
        e_neg_tmc = es_neg_m + ei_neg_m
        e_pos_pc = es_pos_m * gs + ei_pos_m * gi
        e_neg_pc = es_neg_m * gs + ei_neg_m * gi
        e_pos = 0.5 * e_pos_tmc + 0.5 * e_pos_pc
        e_neg = 0.5 * e_neg_tmc + 0.5 * e_neg_pc
        alpha_f, beta_f = e_pos + 1.0, e_neg + 1.0
        p = alpha_f / (alpha_f + beta_f)
        u = 2.0 / (alpha_f + beta_f)
        alpha_s, beta_s = es_pos + 1.0, es_neg + 1.0
        alpha_i, beta_i = ei_pos + 1.0, ei_neg + 1.0
        return p.clamp(1e-6, 1 - 1e-6), {
            "alpha_s": alpha_s, "beta_s": beta_s, "alpha_i": alpha_i, "beta_i": beta_i,
            "alpha_f": alpha_f, "beta_f": beta_f, "gamma_s": gs, "gamma_i": gi,
            "uncertainty": u, "gamma_s_logit": gs_logit, "gamma_i_logit": gi_logit,
        }


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
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p),
            "uncertainty": float(np.concatenate(us).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--p-clean", type=float, default=0.6)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--lambda-unimodal", type=float, default=0.5)
    ap.add_argument("--lambda-gamma", type=float, default=0.05)
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
    model = PCBEFv2Net(sig_tr["clean"][0].shape[1], img_tr["clean"][0].shape[1], q_dim, y_tr.shape[1]).to(device)
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
                lf = evid_elem(aux["alpha_f"], aux["beta_f"], y).mean()
                ls = evid_elem(aux["alpha_s"], aux["beta_s"], y)
                li = evid_elem(aux["alpha_i"], aux["beta_i"], y)
                ls = (ls * ms.squeeze(1)).sum() / (ms.sum() + 1e-6)
                li = (li * mi.squeeze(1)).sum() / (mi.sum() + 1e-6)
                loss = lf + args.lambda_unimodal * (ls + li)
                loss = loss + kl_beta(aux["alpha_s"], aux["beta_s"]) + kl_beta(aux["alpha_i"], aux["beta_i"])
                ps = aux["alpha_s"] / (aux["alpha_s"] + aux["beta_s"])
                pi = aux["alpha_i"] / (aux["alpha_i"] + aux["beta_i"])
                ret_s = 1.0 - (ps - y).abs().mean(dim=1, keepdim=True)
                ret_i = 1.0 - (pi - y).abs().mean(dim=1, keepdim=True)
                tgs = 0.70 + 0.30 * ret_s
                tgi = 0.45 + 0.55 * ret_i
                lg = (F.binary_cross_entropy_with_logits(aux["gamma_s_logit"], ret_s, reduction="none") * ms).sum() / (ms.sum() + 1e-6)
                lg += (F.binary_cross_entropy_with_logits(aux["gamma_i_logit"], ret_i, reduction="none") * mi).sum() / (mi.sum() + 1e-6)
                loss = loss + args.lambda_gamma * lg
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
    rows = [{"sig": cs, "img": ci, **eval_cond(model, sig_te, img_te, y_te, cs, ci, device)} for cs, ci in pairs]
    miss = [{"condition": c, **eval_cond(model, sig_te, img_te, y_te, "clean", "clean", device, mask_s=ms, mask_i=mi)}
            for c, ms, mi in [("signal_only", 1.0, 0.0), ("image_only", 0.0, 1.0)]]
    rng = np.random.default_rng(0)
    qperm = []
    for _ in range(10):
        perm = rng.permutation(len(y_te))
        qperm.append(eval_cond(model, sig_te, img_te, y_te, "clean", "clean", device, q_perm=perm)["auc"])
    clean_auc = [r for r in rows if r["sig"] == "clean" and r["img"] == "clean"][0]["auc"]
    metrics = {"best_epoch": best_epoch, "best_val_robust_auc": best, "rows": rows, "missing": miss,
               "q_perm": {"clean_auc": clean_auc, "perm_mean": float(np.mean(qperm)), "perm_std": float(np.std(qperm)),
                          "delta": float(clean_auc - np.mean(qperm))}, "args": vars(args)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("rows", "args")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

