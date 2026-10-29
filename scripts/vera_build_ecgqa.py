#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vera_train_apeb import APEBNet
from vera_train_pcbef import load_all

GROUPS = ["Inferior", "Septal", "Anterior", "Lateral"]
BUCKET = {"PR": ["short", "normal", "prolonged"], "QRS": ["narrow", "wide"], "QTc": ["short", "normal", "prolonged"]}


def predict_bank(model, zs, zi, device, batch=512):
    ds, dg, dv, db, dr = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(zs), batch):
            a = torch.from_numpy(zs[i:i + batch]).to(device)
            b = torch.from_numpy(zi[i:i + batch]).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(a, b)
            ds.append(torch.sigmoid(out["diag"]).float().cpu())
            dg.append(torch.sigmoid(out["group"]).float().cpu())
            dv.append(torch.sigmoid(out["valid"]).float().cpu())
            db.append(out["bucket"].float().cpu())
            dr.append(out["reg"].float().cpu())
    return (torch.cat(ds).numpy(), torch.cat(dg).numpy(), torch.cat(dv).numpy(),
            torch.cat(db).numpy(), torch.cat(dr).numpy())


def build_anchor(diag, group, valid, bucket, reg_raw, unc):
    def f(x, n=0):
        return f"{x:.{n}f}"
    hr, pr, qrs, qt, qtc = reg_raw
    pr_b = BUCKET["PR"][int(bucket[0:3].argmax())]
    qrs_b = BUCKET["QRS"][int(bucket[3:5].argmax())]
    qtc_b = BUCKET["QTc"][int(bucket[5:8].argmax())]
    diag_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    order = np.argsort(-diag)[:2]
    top = ", ".join(f"{diag_names[i]}={diag[i]:.2f}" for i in order)
    return (
        f"[Evidence | HR: {hr:.0f} bpm (v={valid[0]:.2f}); "
        f"PR: {'Absent' if valid[1] < 0.5 else f'{pr:.0f} ms ({pr_b}, v={valid[1]:.2f})'}; "
        f"QRS: {qrs:.0f} ms ({qrs_b}, v={valid[2]:.2f}); "
        f"QT: {qt:.0f} ms (v={valid[3]:.2f}); QTc: {qtc:.0f} ms ({qtc_b}, v={valid[4]:.2f}) | "
        f"Territory: Inferior={group[0]:.2f}, Septal={group[1]:.2f}, Anterior={group[2]:.2f}, Lateral={group[3]:.2f} | "
        f"Diagnosis: {top}; uncertainty={unc:.2f}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apeb-run", required=True)
    ap.add_argument("--ecgqa-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-template", type=int, default=450)
    ap.add_argument("--test-per-template", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    meta = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    ck = torch.load(os.path.join(args.apeb_run, "best.pt"), map_location=device)
    model = APEBNet(512, 512, 5).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    metrics = json.load(open(os.path.join(args.apeb_run, "metrics.json"), encoding="utf-8"))
    mu = np.asarray(metrics["evidence_mu"], dtype=np.float32)
    sd = np.asarray(metrics["evidence_sd"], dtype=np.float32)

    all_z, all_ev, all_ids, anchors = [], [], [], []
    for split in ["train", "val", "test"]:
        sub = meta[meta["split"] == split].reset_index(drop=True)
        sig, img = load_all(args.feature_root, split, q_dim)
        zs, zi = sig["clean"][0], img["clean"][0]
        diag, group, valid, bucket, reg = predict_bank(model, zs, zi, device)
        reg_raw = reg * sd + mu
        bucket_probs = np.concatenate([
            np.exp(bucket[:, 0:3]) / np.exp(bucket[:, 0:3]).sum(1, keepdims=True),
            np.exp(bucket[:, 3:5]) / np.exp(bucket[:, 3:5]).sum(1, keepdims=True),
            np.exp(bucket[:, 5:8]) / np.exp(bucket[:, 5:8]).sum(1, keepdims=True),
        ], axis=1)
        unc = -(diag * np.log(np.clip(diag, 1e-6, 1)) + (1 - diag) * np.log(np.clip(1 - diag, 1e-6, 1))).mean(1, keepdims=True)
        ev = np.concatenate([reg_raw, group, valid, bucket_probs, diag, unc], axis=1).astype(np.float32)
        z = np.concatenate([zs, zi], axis=1).astype(np.float32)
        for i, eid in enumerate(sub["ecg_id"].values):
            all_z.append(z[i]); all_ev.append(ev[i]); all_ids.append(int(eid))
            anchors.append(build_anchor(diag[i], group[i], valid[i], bucket[i], reg_raw[i], unc[i, 0]))
        print(split, z.shape, ev.shape, flush=True)
    np.savez_compressed(os.path.join(args.out_dir, "ecg_bank.npz"),
                        z=np.stack(all_z), ev=np.stack(all_ev), ecg_id=np.array(all_ids))
    with open(os.path.join(args.out_dir, "anchors.json"), "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False)
    id2idx = {eid: i for i, eid in enumerate(all_ids)}

    def load_split(name, per_tpl):
        files = sorted(glob.glob(os.path.join(args.ecgqa_root, "template", name, "*.json")))
        by_tpl = defaultdict(list)
        for fn in files:
            data = json.load(open(fn, encoding="utf-8"))
            for s in data:
                ids = [int(x) for x in s["ecg_id"]]
                if all(x in id2idx for x in ids):
                    by_tpl[s["template_id"]].append(s)
        out = []
        for tid, items in by_tpl.items():
            out.extend(items[:per_tpl])
        print(name, "selected", len(out), "templates", len(by_tpl), flush=True)
        return out

    train = load_split("train", args.per_template)
    test = load_split("test", args.test_per_template)
    for name, items in [("train", train), ("test", test)]:
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for s in items:
                rec = {
                    "question_type": s["question_type"], "attribute_type": s["attribute_type"],
                    "question": s["question"], "answer": s["answer"],
                    "ecg_ids": [int(x) for x in s["ecg_id"]], "template_id": s["template_id"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "id2idx.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2idx.items()}, f)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
