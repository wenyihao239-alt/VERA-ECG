#!/usr/bin/env python3
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from train_signal import ece, macro_ap, macro_auc, macro_f1
from vera_train_pcbef import PCBEFNet, load_all

PAIRS = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
         ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
         ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30")]


def build_mixed(sig, img, y, pairs):
    zs = np.concatenate([sig[cs][0] for cs, ci in pairs], axis=0)
    qs = np.concatenate([sig[cs][1] for cs, ci in pairs], axis=0)
    zi = np.concatenate([img[ci][0] for cs, ci in pairs], axis=0)
    qi = np.concatenate([img[ci][1] for cs, ci in pairs], axis=0)
    yy = np.concatenate([y for _ in pairs], axis=0)
    return zs, qs, zi, qi, yy


def tmc_outputs(model, zs, zi, device):
    k = model.num_classes
    out = []
    with torch.no_grad():
        for i in range(0, len(zs), 512):
            a = torch.from_numpy(zs[i:i + 512]).to(device)
            b = torch.from_numpy(zi[i:i + 512]).to(device)
            rs, ri = model.ev_s(a), model.ev_i(b)
            es_pos = F.softplus(rs[:, :k]) + 1e-6
            es_neg = F.softplus(rs[:, k:]) + 1e-6
            ei_pos = F.softplus(ri[:, :k]) + 1e-6
            ei_neg = F.softplus(ri[:, k:]) + 1e-6
            out.append(torch.cat([es_pos, es_neg, ei_pos, ei_neg], dim=1).cpu())
    return torch.cat(out, dim=0)


def v1_gamma(model, qs, qi, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(qs), 512):
            a = torch.from_numpy(qs[i:i + 512]).to(device)
            b = torch.from_numpy(qi[i:i + 512]).to(device)
            gs = torch.sigmoid(model.gamma_s(a))
            gi = torch.sigmoid(model.gamma_i(b))
            out.append(torch.cat([gs, gi], dim=1).cpu())
    return torch.cat(out, dim=0)


def ds_fuse(es_pos, es_neg, ei_pos, ei_neg):
    s_pos, s_neg = es_pos + 1.0, es_neg + 1.0
    i_pos, i_neg = ei_pos + 1.0, ei_neg + 1.0
    bs_p, bs_n = es_pos / (s_pos + s_neg), es_neg / (s_pos + s_neg)
    bi_p, bi_n = ei_pos / (i_pos + i_neg), ei_neg / (i_pos + i_neg)
    us, ui = 2.0 / (s_pos + s_neg), 2.0 / (i_pos + i_neg)
    K = bs_p * bi_n + bs_n * bi_p
    den = (1.0 - K).clamp_min(1e-4)
    bp = (bs_p * bi_p + bs_p * ui + us * bi_p) / den
    bn = (bs_n * bi_n + bs_n * ui + us * bi_n) / den
    u = (us * ui) / den
    norm = (bp + bn + u).clamp_min(1e-6)
    bp, bn, u = bp / norm, bn / norm, u / norm
    p = (bp + 0.5 * u).clamp(1e-6, 1 - 1e-6)
    return p, u, K


def metrics_from_p(y, p):
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p)}


def aurc(y, p):
    pp = np.clip(p, 1e-6, 1 - 1e-6)
    ent = -(pp * np.log(pp) + (1 - pp) * np.log(1 - pp)).mean(axis=1) / np.log(2)
    conf = 1 - ent
    err = 1 - ((p >= 0.5) == y).mean(axis=1)
    order = np.argsort(-conf)
    grid = np.linspace(0.05, 1.0, 20)
    risks = []
    for c in grid:
        k = max(1, int(c * len(y)))
        risks.append(float(err[order[:k]].mean()))
    return float(np.mean(risks))


def eval_params(ev, gamma, y, tau_s, tau_i, alpha, device="cpu"):
    es_pos, es_neg, ei_pos, ei_neg = ev[:, :5], ev[:, 5:10], ev[:, 10:15], ev[:, 15:20]
    gs, gi = gamma[:, 0:1], gamma[:, 1:2]
    bs_p, bs_n = es_pos / (es_pos + es_neg + 2), es_neg / (es_pos + es_neg + 2)
    bi_p, bi_n = ei_pos / (ei_pos + ei_neg + 2), ei_neg / (ei_pos + ei_neg + 2)
    C = bs_p * bi_n + bs_n * bi_p
    gs_e = gs ** (tau_s * (1.0 + alpha * C))
    gi_e = gi ** (tau_i * (1.0 + alpha * C))
    p, u, K = ds_fuse(es_pos * gs_e, es_neg * gs_e, ei_pos * gi_e, ei_neg * gi_e)
    p = p.numpy()
    if not isinstance(y, np.ndarray):
        y = y.numpy()
    return {"auc": macro_auc(y, p), "ece": ece(y, p), "aurc": aurc(y, p), "p": p, "u": u.numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tmc-run", required=True)
    ap.add_argument("--v1-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig_va, img_va = load_all(args.feature_root, "val", q_dim)
    sig_te, img_te = load_all(args.feature_root, "test", q_dim)
    y_va = np.load(os.path.join(args.data_dir, "y_val_super.npy"))
    y_te = np.load(os.path.join(args.data_dir, "y_test_super.npy"))
    ck_t = torch.load(os.path.join(args.tmc_run, "best.pt"), map_location=device)
    ck_v = torch.load(os.path.join(args.v1_run, "best.pt"), map_location=device)
    d_s, d_i = sig_va["clean"][0].shape[1], img_va["clean"][0].shape[1]
    tmc = PCBEFNet(d_s, d_i, q_dim, y_va.shape[1], mode="tmc").to(device)
    tmc.load_state_dict(ck_t["model"]); tmc.eval()
    v1 = PCBEFNet(d_s, d_i, q_dim, y_va.shape[1], mode="pcbef").to(device)
    v1.load_state_dict(ck_v["model"]); v1.eval()

    zs_va, qs_va, zi_va, qi_va, yy_va = build_mixed(sig_va, img_va, y_va, PAIRS)
    ev_va = tmc_outputs(tmc, zs_va, zi_va, device)
    g_va = v1_gamma(v1, qs_va, qi_va, device)
    zs_te, qs_te, zi_te, qi_te, yy_te = build_mixed(sig_te, img_te, y_te, PAIRS)
    ev_te = tmc_outputs(tmc, zs_te, zi_te, device)
    g_te = v1_gamma(v1, qs_te, qi_te, device)

    best = None
    for tau_s, tau_i, alpha in itertools.product([0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
                                                 [0.50, 0.60, 0.70, 0.80, 0.90],
                                                 [0.0, 0.25, 0.5, 0.75, 1.0]):
        m = eval_params(ev_va, g_va, yy_va, tau_s, tau_i, alpha)
        if best is None or m["auc"] > best["val_auc"]:
            best = {"tau_s": tau_s, "tau_i": tau_i, "alpha": alpha, "val_auc": m["auc"], "val_ece": m["ece"]}
    tau_s, tau_i, alpha = best["tau_s"], best["tau_i"], best["alpha"]
    base = eval_params(ev_te, g_te, yy_te, tau_s, tau_i, alpha)
    # q/gamma permutation
    rng = np.random.default_rng(0)
    q_aucs, q_eces, q_aurcs = [], [], []
    g_aucs, g_eces, g_aurcs = [], [], []
    for _ in range(10):
        perm = rng.permutation(len(yy_te))
        g_q = v1_gamma(v1, qs_te[perm], qi_te[perm], device)
        m = eval_params(ev_te, g_q, yy_te, tau_s, tau_i, alpha)
        q_aucs.append(m["auc"]); q_eces.append(m["ece"]); q_aurcs.append(m["aurc"])
        m = eval_params(ev_te, g_te[perm], yy_te, tau_s, tau_i, alpha)
        g_aucs.append(m["auc"]); g_eces.append(m["ece"]); g_aurcs.append(m["aurc"])
    out = {
        "best": best, "clean_mixed": {k: base[k] for k in ["auc", "ece", "aurc"]},
        "q_perm": {"auc_mean": float(np.mean(q_aucs)), "auc_delta": float(base["auc"] - np.mean(q_aucs)),
                   "ece_mean": float(np.mean(q_eces)), "ece_delta": float(base["ece"] - np.mean(q_eces)),
                   "aurc_mean": float(np.mean(q_aurcs)), "aurc_delta": float(base["aurc"] - np.mean(q_aurcs))},
        "gamma_perm": {"auc_mean": float(np.mean(g_aucs)), "auc_delta": float(base["auc"] - np.mean(g_aucs)),
                       "ece_mean": float(np.mean(g_eces)), "ece_delta": float(base["ece"] - np.mean(g_eces)),
                       "aurc_mean": float(np.mean(g_aurcs)), "aurc_delta": float(base["aurc"] - np.mean(g_aurcs))},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


