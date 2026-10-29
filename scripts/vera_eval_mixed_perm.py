#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_signal import macro_auc
from vera_train_pcbef import PCBEFNet, load_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig, img = load_all(args.feature_root, "test", q_dim)
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    ck = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    model = PCBEFNet(sig["clean"][0].shape[1], img["clean"][0].shape[1], q_dim, y.shape[1], mode=ck["args"]["mode"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    pairs = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
             ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
             ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30")]
    zs = np.concatenate([sig[cs][0] for cs, ci in pairs], axis=0)
    qs = np.concatenate([sig[cs][1] for cs, ci in pairs], axis=0)
    zi = np.concatenate([img[ci][0] for cs, ci in pairs], axis=0)
    qi = np.concatenate([img[ci][1] for cs, ci in pairs], axis=0)
    yy = np.concatenate([y for _ in pairs], axis=0)
    zs_t = torch.from_numpy(zs).to(device)
    zi_t = torch.from_numpy(zi).to(device)
    qs_t = torch.from_numpy(qs).to(device)
    qi_t = torch.from_numpy(qi).to(device)

    def predict(qs_x=None, qi_x=None, gamma=None):
        q1 = qs_t if qs_x is None else qs_x
        q2 = qi_t if qi_x is None else qi_x
        ps = []
        with torch.no_grad():
            for i in range(0, len(zs_t), 512):
                gb = None if gamma is None else (gamma[0][i:i + 512], gamma[1][i:i + 512])
                p, _ = model(zs_t[i:i + 512], zi_t[i:i + 512], q1[i:i + 512], q2[i:i + 512], gamma_override=gb)
                ps.append(p.float().cpu().numpy())
        return np.concatenate(ps, axis=0)

    base_p = predict()
    base_auc = macro_auc(yy, base_p)
    gs = torch.sigmoid(model.gamma_s(qs_t))
    gi = torch.sigmoid(model.gamma_i(qi_t))
    rng = np.random.default_rng(0)
    q_aucs, g_aucs = [], []
    for _ in range(10):
        perm = rng.permutation(len(yy))
        qs_p = torch.from_numpy(qs[perm]).to(device)
        qi_p = torch.from_numpy(qi[perm]).to(device)
        q_aucs.append(macro_auc(yy, predict(qs_p, qi_p)))
        g_aucs.append(macro_auc(yy, predict(gamma=(gs[perm], gi[perm]))))
    out = {
        "pairs": pairs,
        "n": int(len(yy)),
        "clean_mixed_auc": float(base_auc),
        "q_perm_mean": float(np.mean(q_aucs)),
        "q_perm_std": float(np.std(q_aucs)),
        "q_perm_delta": float(base_auc - np.mean(q_aucs)),
        "gamma_perm_mean": float(np.mean(g_aucs)),
        "gamma_perm_std": float(np.std(g_aucs)),
        "gamma_perm_delta": float(base_auc - np.mean(g_aucs)),
        "gamma_s_mean": float(gs.mean()),
        "gamma_i_mean": float(gi.mean()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

