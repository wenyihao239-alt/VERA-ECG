#!/usr/bin/env python3
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DATA = os.path.join(BASE, "data", "processed", "ptbxl_100")
RENDER = os.path.join(BASE, "data", "rendered", "ptbxl_320")
IQ = os.path.join(BASE, "features", "image_quality")
LOG = os.path.join(BASE, "logs", "multiseed.log")


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


def main():
    for seed in [43, 44]:
        a0 = os.path.join(BASE, "runs", f"A0_resnet1d_seed{seed}")
        a1 = os.path.join(BASE, "runs", f"A1_resnet18_seed{seed}")
        fs = os.path.join(BASE, "features", f"A0_resnet1d_seed{seed}")
        fi = os.path.join(BASE, "features", f"A1_resnet18_seed{seed}")
        run([PY, os.path.join(BASE, "scripts", "train_signal.py"),
             "--data-dir", DATA, "--task", "super", "--model", "resnet1d",
             "--epochs", "100", "--batch-size", "128", "--lr", "1e-3",
             "--seed", str(seed), "--workers", "6", "--device", "cuda:0", "--out-dir", a0], f"A0 seed{seed}")
        run([PY, os.path.join(BASE, "scripts", "train_image.py"),
             "--data-dir", DATA, "--render-dir", RENDER, "--task", "super", "--model", "resnet18",
             "--image-size", "224", "--epochs", "50", "--batch-size", "64", "--lr", "3e-4",
             "--seed", str(seed), "--workers", "6", "--device", "cuda:0", "--out-dir", a1], f"A1 seed{seed}")
        run([PY, os.path.join(BASE, "scripts", "extract_features.py"),
             "--modality", "signal", "--data-dir", DATA, "--task", "super", "--num-classes", "5",
             "--model", "resnet1d", "--checkpoint", os.path.join(a0, "best.pt"),
             "--out-dir", fs, "--batch-size", "256", "--workers", "6", "--device", "cuda:0"], f"feat signal seed{seed}")
        run([PY, os.path.join(BASE, "scripts", "extract_features.py"),
             "--modality", "image", "--data-dir", DATA, "--render-dir", RENDER, "--task", "super",
             "--num-classes", "5", "--model", "resnet18", "--image-size", "224",
             "--checkpoint", os.path.join(a1, "best.pt"), "--out-dir", fi,
             "--batch-size", "128", "--workers", "6", "--device", "cuda:0"], f"feat image seed{seed}")
        for mode in ["gated_md", "quality"]:
            run([PY, os.path.join(BASE, "scripts", "train_fusion.py"),
                 "--data-dir", DATA, "--signal-features", fs, "--image-features", fi, "--image-quality", IQ,
                 "--task", "super", "--mode", mode, "--epochs", "100", "--batch-size", "512", "--lr", "1e-3",
                 "--seed", str(seed), "--workers", "0", "--device", "cuda:0",
                 "--out-dir", os.path.join(BASE, "runs", f"fusion_{mode}_seed{seed}")], f"fusion {mode} seed{seed}")
    log("multiseed done")


if __name__ == "__main__":
    main()
