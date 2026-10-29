#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_fusion import FusionNet, normalize
from train_signal import macro_ap, macro_auc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--signal-features", required=True)
    ap.add_argument("--image-features", required=True)
    ap.add_argument("--image-quality", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    zs = {s: np.load(os.path.join(args.signal_features, f"signal_{s}.npy")) for s in ["train", "val", "test"]}
    zi = {s: np.load(os.path.join(args.image_features, f"image_{s}.npy")) for s in ["train", "val", "test"]}
    qs = {s: np.load(os.path.join(args.data_dir, f"q_{s}.npy")) for s in ["train", "val", "test"]}
    qi = {s: np.load(os.path.join(args.image_quality, f"qi_{s}.npy")) for s in ["train", "val", "test"]}
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    zs["train"], zs["val"], zs["test"] = normalize(zs["train"], zs["val"], zs["test"])
    zi["train"], zi["val"], zi["test"] = normalize(zi["train"], zi["val"], zi["test"])
    qs["train"], qs["val"], qs["test"] = normalize(qs["train"], qs["val"], qs["test"])
    q_dim = qs["train"].shape[1]
    for s in ["train", "val", "test"]:
        if qi[s].shape[1] < q_dim:
            qi[s] = np.concatenate([qi[s], np.zeros((len(qi[s]), q_dim - qi[s].shape[1]), dtype=np.float32)], axis=1)
        elif qi[s].shape[1] > q_dim:
            qi[s] = qi[s][:, :q_dim]
    qi["train"], qi["val"], qi["test"] = normalize(qi["train"], qi["val"], qi["test"])

    ck = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    mode = ck["args"]["mode"]
    model = FusionNet(mode, zs["test"].shape[1], zi["test"].shape[1], q_dim, y.shape[1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    zs_t = torch.from_numpy(zs["test"]).to(device)
    zi_t = torch.from_numpy(zi["test"]).to(device)
    qs_np = qs["test"]
    qi_np = qi["test"]

    def run(qs_row, qi_row):
        qs_t = torch.from_numpy(qs_row).to(device)
        qi_t = torch.from_numpy(qi_row).to(device)
        with torch.no_grad():
            logits, aux = model(zs_t, zi_t, qs_t, qi_t)
            p = torch.sigmoid(logits).float().cpu().numpy()
        return macro_auc(y, p), macro_ap(y, p), macro_f1(y, p, 0.5), float(aux["gate"].mean().cpu())

    base = run(qs_np, qi_np)
    rng = np.random.default_rng(0)
    rows = [{"perm": -1, "auc": base[0], "ap": base[1], "f1": base[2], "gate_mean": base[3]}]
    for k in range(args.n_perm):
        perm = rng.permutation(len(y))
        m = run(qs_np[perm], qi_np[perm])
        rows.append({"perm": k, "auc": m[0], "ap": m[1], "f1": m[2], "gate_mean": m[3]})
        print(rows[-1], flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    aucs = [r["auc"] for r in rows[1:]]
    print(json.dumps({"clean_auc": base[0], "perm_auc_mean": float(np.mean(aucs)), "perm_auc_std": float(np.std(aucs)), "delta": float(base[0] - np.mean(aucs))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
