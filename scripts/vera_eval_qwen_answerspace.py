#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from vera_train_qwen import Projector, SPECIAL
from vera_eval_qwen import build_prompt, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--answers-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=350)
    ap.add_argument("--max-cands", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    tok_ids = {s: tok.convert_tokens_to_ids(s) for s in SPECIAL}
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    projector = Projector(1052, 1024, 8, 2048).to(device)
    projector.load_state_dict(torch.load(args.projector, map_location=device))
    projector.eval()
    emb_layer = model.get_input_embeddings()

    bank = np.load(args.bank)
    anchors = json.load(open(os.path.join(args.qa_dir, "anchors.json"), encoding="utf-8"))
    id2idx = json.load(open(os.path.join(args.qa_dir, "id2idx.json"), encoding="utf-8"))
    cmap = {}
    with open(args.answers_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cmap[int(row["template_id"])] = ast.literal_eval(row["classes"])
    items_all = [json.loads(x) for x in open(os.path.join(args.qa_dir, "test.jsonl"), encoding="utf-8")]
    groups = defaultdict(list)
    for x in items_all:
        groups[x["question_type"]].append(x)
    per = max(1, args.samples // max(1, len(groups)))
    items = []
    for k, v in groups.items():
        random.seed(0); random.shuffle(v)
        items.extend(v[:per])

    out, correct, total = [], 0, 0
    by = {}
    for s in items:
        cands = cmap.get(s["template_id"], [])
        if not cands or len(cands) > args.max_cands:
            continue
        idxs = [id2idx[str(x)] for x in s["ecg_ids"]]
        a = bank["z"][idxs[0]].astype(np.float32); ea = bank["ev"][idxs[0]].astype(np.float32)
        b = np.zeros_like(a); eb = np.zeros_like(ea); d = np.zeros_like(a); ed = np.zeros_like(ea)
        two = len(idxs) > 1
        if two:
            b = bank["z"][idxs[1]].astype(np.float32); eb = bank["ev"][idxs[1]].astype(np.float32)
            d = b - a; ed = eb - ea
        prompt = build_prompt(s, anchors[idxs[0]], anchors[idxs[1]] if two else "", two)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids], dtype=torch.long, device=device)
        emb = emb_layer(ids).float()
        sa = projector(torch.from_numpy(np.concatenate([a, ea]))[None].to(device))
        sb = projector(torch.from_numpy(np.concatenate([b, eb]))[None].to(device))
        sd = projector(torch.from_numpy(np.concatenate([d, ed]))[None].to(device))
        for name, soft in [("<ECG_A>", sa), ("<ECG_B>", sb), ("<ECG_D>", sd)]:
            pos = (ids[0] == tok_ids[name]).nonzero(as_tuple=False).flatten()
            if len(pos):
                emb[0, pos] = soft[0, :len(pos)].to(emb.dtype)
        emb = emb.half()
        losses = []
        with torch.no_grad():
            for c in cands:
                cid = tok(c + "<|im_end|>", add_special_tokens=False)["input_ids"]
                ce = emb_layer(torch.tensor([cid], dtype=torch.long, device=device)).half()
                full = torch.cat([emb, ce], dim=1)
                labels = torch.full((1, full.size(1)), -100, dtype=torch.long, device=device)
                labels[0, emb.size(1):] = torch.tensor(cid, dtype=torch.long, device=device)
                loss = model(inputs_embeds=full, labels=labels).loss
                losses.append(float(loss))
        pred = cands[int(np.argmin(losses))]
        gold = [norm(x) for x in (s["answer"] if isinstance(s["answer"], list) else [str(s["answer"])])]
        ok = norm(pred) in gold
        correct += int(ok); total += 1
        qt = s["question_type"]
        by.setdefault(qt, [0, 0]); by[qt][0] += int(ok); by[qt][1] += 1
        out.append({"question": s["question"], "cands": cands, "pred": pred, "gold": s["answer"], "ok": ok, "type": qt})
    res = {"n": total, "accuracy": correct / max(1, total),
           "by_type": {k: {"correct": v[0], "n": v[1], "acc": v[0] / max(1, v[1])} for k, v in by.items()},
           "examples": out[:30]}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
