#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DATA = os.path.join(BASE, "data", "processed", "ptbxl_100")
ROOT = os.path.join(BASE, "data", "corrupted_features")
CKPT_A0 = os.path.join(BASE, "runs", "A0_resnet1d_seed42", "best.pt")
CKPT_A1 = os.path.join(BASE, "runs", "A1_resnet18_seed42", "best.pt")
FEAT_S = os.path.join(BASE, "features", "A0_resnet1d_seed42")
FEAT_I = os.path.join(BASE, "features", "A1_resnet18_seed42")
IQ = os.path.join(BASE, "features", "image_quality")
RENDER = os.path.join(BASE, "data", "rendered", "ptbxl_320")
OUT = os.path.join(BASE, "runs_corrupted")
LOG = os.path.join(BASE, "logs", "corruption_pipeline.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, name):
    log("START " + name)
    t = time.time()
    p = subprocess.run(cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(p.stdout)
    if p.returncode != 0:
        log(f"FAIL {name} rc={p.returncode}")
        raise SystemExit(p.returncode)
    log(f"DONE {name} {time.time()-t:.1f}s")


def copy_clean():
    for split in ["train", "val", "test"]:
        d = os.path.join(ROOT, "signal", "clean")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(FEAT_S, f"signal_{split}.npy"), os.path.join(d, f"signal_{split}.npy"))
        shutil.copy2(os.path.join(DATA, f"q_{split}.npy"), os.path.join(d, f"q_{split}.npy"))
        d = os.path.join(ROOT, "image", "clean")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(FEAT_I, f"image_{split}.npy"), os.path.join(d, f"image_{split}.npy"))
        shutil.copy2(os.path.join(IQ, f"qi_{split}.npy"), os.path.join(d, f"qi_{split}.npy"))
    log("clean features copied")


def main():
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log("pipeline start")
    copy_clean()

    sig_conds = ["noise10", "baseline03", "leaddrop3"]
    for cond in sig_conds:
        done = all(os.path.exists(os.path.join(ROOT, "signal", cond, f"signal_{s}.npy")) for s in ["train", "val", "test"])
        if done:
            log(f"skip signal {cond}")
            continue
        run([PY, os.path.join(BASE, "scripts", "extract_corrupted_features.py"),
             "--modality", "signal", "--condition", cond,
             "--data-dir", DATA, "--checkpoint", CKPT_A0, "--model", "resnet1d",
             "--num-classes", "5", "--out-dir", ROOT, "--batch-size", "256",
             "--chunk", "512", "--workers", "0", "--device", "cuda:0"], f"signal {cond}")

    img_conds = ["blur2", "crop10", "jpeg30"]
    for cond in img_conds:
        done = all(os.path.exists(os.path.join(ROOT, "image", cond, f"image_{s}.npy")) for s in ["train", "val", "test"])
        if done:
            log(f"skip image {cond}")
            continue
        run([PY, os.path.join(BASE, "scripts", "extract_corrupted_features.py"),
             "--modality", "image", "--condition", cond,
             "--data-dir", DATA, "--render-dir", RENDER, "--checkpoint", CKPT_A1,
             "--model", "resnet18", "--num-classes", "5", "--image-size", "224",
             "--out-dir", ROOT, "--batch-size", "128", "--workers", "4", "--device", "cuda:0"], f"image {cond}")

    for mode in ["gated_md", "quality"]:
        out = os.path.join(OUT, f"{mode}_seed42")
        run([PY, os.path.join(BASE, "scripts", "train_fusion_corrupted.py"),
             "--feature-root", ROOT, "--data-dir", DATA, "--task", "super",
             "--mode", mode, "--epochs", "100", "--batch-size", "512", "--lr", "1e-3",
             "--seed", "42", "--patience", "15", "--device", "cuda:0", "--out-dir", out], f"train {mode}")

    summary = {}
    for mode in ["gated_md", "quality"]:
        p = os.path.join(OUT, f"{mode}_seed42", "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            summary[mode] = {
                "clean_auc": m["clean"]["auc"],
                "signal_corrupted_mean_auc": m["signal_corrupted_mean_auc"],
                "image_corrupted_mean_auc": m["image_corrupted_mean_auc"],
                "robust_mean_auc": m["robust_mean_auc"],
                "best_epoch": m["best_epoch"],
            }
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    with open(os.path.join(BASE, "results", "corruption_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("pipeline done " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
