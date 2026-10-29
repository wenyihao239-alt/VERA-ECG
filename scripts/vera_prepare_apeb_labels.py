#!/usr/bin/env python3
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

GROUPS = {
    "Inferior": ["IMI", "IPLMI", "IPMI", "ISCI", "ISCIN", "ISCIL", "ISCLA"],
    "Septal": ["ASMI", "CRBBB", "IRBBB", "RVH"],
    "Anterior": ["AMI", "ISCA", "ISCAN", "ANEUR"],
    "Lateral": ["LMI", "ALMI", "ISCL", "LVH", "CLBBB", "ILBBB"],
}
LEAD_GROUP = {
    "II": "Inferior", "III": "Inferior", "aVF": "Inferior",
    "V1": "Septal", "V2": "Septal",
    "V3": "Anterior", "V4": "Anterior",
    "I": "Lateral", "aVL": "Lateral", "V5": "Lateral", "V6": "Lateral",
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--raw-csv", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    raw = pd.read_csv(args.raw_csv, usecols=["ecg_id", "scp_codes"])
    raw["codes"] = raw["scp_codes"].apply(parse)
    code_map = dict(zip(raw["ecg_id"], raw["codes"]))
    groups = list(GROUPS.keys())
    for split in ["train", "val", "test"]:
        sub = meta[meta["split"] == split].reset_index(drop=True)
        g = np.zeros((len(sub), 4), dtype=np.float32)
        for i, eid in enumerate(sub["ecg_id"].values):
            codes = code_map.get(int(eid), {})
            for j, name in enumerate(groups):
                if any(c in codes for c in GROUPS[name]):
                    g[i, j] = 1.0
        # 12-lead labels expanded from groups (weak, zero-cost)
        lead12 = np.zeros((len(sub), 12), dtype=np.float32)
        for j, lead in enumerate(LEADS):
            gname = LEAD_GROUP.get(lead)
            if gname in groups:
                lead12[:, j] = g[:, groups.index(gname)]
        np.save(os.path.join(args.out_dir, f"lead_group_{split}.npy"), g)
        np.save(os.path.join(args.out_dir, f"lead12_{split}.npy"), lead12)
        print(split, "group counts", g.sum(axis=0), flush=True)
    with open(os.path.join(args.out_dir, "apeb_names.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": groups, "leads": LEADS}, f, ensure_ascii=False, indent=2)

    # bucket labels for PR/QRS/QTc
    for split in ["train", "val", "test"]:
        ev = np.load(os.path.join(args.evidence_dir, f"evidence_{split}.npy"))
        mask = np.load(os.path.join(args.evidence_dir, f"evidence_mask_{split}.npy"))
        bucket = np.full((len(ev), 3), -1, dtype=np.int64)
        pr, qrs, qtc = ev[:, 1], ev[:, 2], ev[:, 4]
        mpr, mqrs, mqtc = mask[:, 1] > 0, mask[:, 2] > 0, mask[:, 4] > 0
        bucket[mpr & (pr < 120), 0] = 0
        bucket[mpr & (pr >= 120) & (pr <= 200), 0] = 1
        bucket[mpr & (pr > 200), 0] = 2
        bucket[mqrs & (qrs < 120), 1] = 0
        bucket[mqrs & (qrs >= 120), 1] = 1
        bucket[mqtc & (qtc < 350), 2] = 0
        bucket[mqtc & (qtc >= 350) & (qtc <= 450), 2] = 1
        bucket[mqtc & (qtc > 450), 2] = 2
        np.save(os.path.join(args.out_dir, f"bucket_{split}.npy"), bucket)
        np.save(os.path.join(args.out_dir, f"bucket_mask_{split}.npy"), (bucket >= 0).astype(np.float32))
        print(split, "bucket valid", (bucket >= 0).mean(axis=0), flush=True)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
