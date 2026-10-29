#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_signal_mv import ResNet1dMV
from vera_prepare_mv import stats_for_split

GROUPS = {"Inferior": [1, 2, 5], "Septal": [6, 7], "Anterior": [8, 9], "Lateral": [0, 4, 10, 11]}
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]


def predict(model, x, mv_mean, mv_std, device):
    raw = x[None].astype(np.float32)
    mv = stats_for_split(raw)
    mu = raw.mean(axis=2, keepdims=True); sd = raw.std(axis=2, keepdims=True) + 1e-6
    xb = (raw - mu) / sd
    mv = (mv - mv_mean) / mv_std
    with torch.no_grad():
        xb = torch.from_numpy(xb).to(device); mb = torch.from_numpy(mv.astype(np.float32)).to(device)
        logit = model(xb, mb).float().cpu().numpy()[0]
        p = 1.0 / (1.0 + np.exp(-logit))
    ent = -(p * np.log(np.clip(p, 1e-6, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-6, 1))).mean()
    return p, float(ent), logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apeb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    model = ResNet1dMV(num_classes=5, mv_dim=36)
    model.load_state_dict(ck["model"]); model.to(device).eval()
    mv_mean = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_mean.npy")).reshape(-1)
    mv_std = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_std.npy")).reshape(-1)
    x = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y = np.load(os.path.join(args.data_dir, "y_test_super.npy"))
    g = np.load(os.path.join(args.apeb_dir, "lead_group_test.npy"))
    rows = []
    rng = np.random.default_rng(0)
    for i in range(len(y)):
        if len(rows) >= args.max_samples:
            break
        pos = np.where(g[i] > 0.5)[0]
        if len(pos) != 1:
            continue
        gname = list(GROUPS.keys())[pos[0]]
        target_cls = 4 if gname == "Lateral" else 1
        if y[i, target_cls] < 0.5:
            continue
        p0, e0, l0 = predict(model, x[i], mv_mean, mv_std, device)
        xt = x[i].copy()
        xt[GROUPS[gname]] = 0
        pt, et, lt = predict(model, xt, mv_mean, mv_std, device)
        others = [k for k in GROUPS if k != gname]
        rname = others[rng.integers(len(others))]
        xr = x[i].copy()
        lead_idx = GROUPS[rname][:len(GROUPS[gname])]
        xr[lead_idx] = 0
        pr, er, lr = predict(model, xr, mv_mean, mv_std, device)
        rows.append({"i": int(i), "group": gname, "random": rname, "target_cls": SUPER[target_cls],
                     "p0": float(p0[target_cls]), "p_target": float(pt[target_cls]), "p_random": float(pr[target_cls]),
                     "d_target": float(p0[target_cls] - pt[target_cls]), "d_random": float(p0[target_cls] - pr[target_cls]),
                     "entropy0": e0, "entropy_target": et, "entropy_random": er,
                     "logit0": float(l0[target_cls]), "logit_target": float(lt[target_cls]), "logit_random": float(lr[target_cls]),
                     "d_logit_target": float(l0[target_cls] - lt[target_cls]), "d_logit_random": float(l0[target_cls] - lr[target_cls])})
    dt = np.mean([r["d_target"] for r in rows]); dr = np.mean([r["d_random"] for r in rows])
    out = {"n": len(rows), "mean_drop_target": float(dt), "mean_drop_random": float(dr),
           "csr": float(dt / dr) if dr != 0 else float("inf"),
           "mean_entropy_increase_target": float(np.mean([r["entropy_target"] - r["entropy0"] for r in rows])),
           "mean_entropy_increase_random": float(np.mean([r["entropy_random"] - r["entropy0"] for r in rows])),
           "mean_logit_drop_target": float(np.mean([r["d_logit_target"] for r in rows])),
           "mean_logit_drop_random": float(np.mean([r["d_logit_random"] for r in rows])),
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



