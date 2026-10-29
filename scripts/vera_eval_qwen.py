#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from vera_train_qwen import Projector, SPECIAL


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_prompt(s, anchor_a, anchor_b, two):
    if two:
        return (
            "<|im_start|>system\nYou are an ECG interpretation assistant. Use the evidence anchors and delta evidence.\n<|im_end|>\n"
            "<|im_start|>user\n"
            "ECG_A: " + "<ECG_A>" * 8 + "\n"
            "ECG_B: " + "<ECG_B>" * 8 + "\n"
            "Delta: " + "<ECG_D>" * 8 + "\n"
            f"Evidence A: {anchor_a}\nEvidence B: {anchor_b}\n"
            f"Question: {s['question']}\n<|im_end|>\n<|im_start|>assistant\n"
        )
    return (
        "<|im_start|>system\nYou are an ECG interpretation assistant. Use the evidence anchor.\n<|im_end|>\n"
        "<|im_start|>user\n"
        "ECG: " + "<ECG_A>" * 8 + "\n"
        f"Evidence: {anchor_a}\n"
        f"Question: {s['question']}\n<|im_end|>\n<|im_start|>assistant\n"
    )


def match_candidates(text, candidates):
    remaining = " " + norm(text) + " "
    found = []
    for c in sorted(candidates, key=lambda x: -len(norm(x))):
        nc = norm(c)
        if not nc:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(nc) + r"(?![a-z0-9])", remaining):
            found.append(nc)
            remaining = remaining.replace(" " + nc + " ", " ", 1)
    return found


def split_pred(text):
    t = norm(text)
    parts = [norm(x) for x in re.split(r"[;\n]", t)]
    return [x for x in parts if x]


def set_scores(pred_text, gold, candidates):
    gold_set = {norm(x) for x in (gold if isinstance(gold, list) else [gold]) if norm(x)}
    matched = match_candidates(pred_text, candidates) if candidates else []
    pred_set = set(matched) if matched else set(split_pred(pred_text))
    em = float(pred_set == gold_set)
    inter = len(pred_set & gold_set)
    if not pred_set and not gold_set:
        f1 = 1.0
    elif not pred_set or not gold_set:
        f1 = 0.0
    else:
        f1 = 2.0 * inter / (len(pred_set) + len(gold_set))
    return em, f1, sorted(pred_set), sorted(gold_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--answers-csv", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
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
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    projector = Projector(1052, 1024, 8, 2048).to(device)
    projector.load_state_dict(torch.load(args.projector, map_location=device))
    projector.eval()

    bank = np.load(args.bank)
    anchors = json.load(open(os.path.join(args.qa_dir, "anchors.json"), encoding="utf-8"))
    id2idx = json.load(open(os.path.join(args.qa_dir, "id2idx.json"), encoding="utf-8"))
    cmap = {}
    if args.answers_csv and os.path.exists(args.answers_csv):
        with open(args.answers_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cmap[int(row["template_id"])] = ast.literal_eval(row["classes"])
    items_all = [json.loads(x) for x in open(os.path.join(args.qa_dir, "test.jsonl"), encoding="utf-8")]
    if args.stratified and not args.all:
        groups = defaultdict(list)
        for x in items_all:
            groups[x["question_type"]].append(x)
        per = max(1, args.samples // max(1, len(groups)))
        items = []
        for k, v in groups.items():
            random.seed(0)
            random.shuffle(v)
            items.extend(v[:per])
    elif args.all:
        items = items_all
    else:
        items = items_all[:args.samples]
    out = []
    legacy_correct = 0
    set_em_sum = 0.0
    set_f1_sum = 0.0
    by_type = {}
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for k, s in enumerate(items):
        idxs = [id2idx[str(x)] for x in s["ecg_ids"]]
        a = bank["z"][idxs[0]].astype(np.float32)
        ea = bank["ev"][idxs[0]].astype(np.float32)
        b = np.zeros_like(a)
        eb = np.zeros_like(ea)
        d = np.zeros_like(a)
        ed = np.zeros_like(ea)
        two = len(idxs) > 1
        if two:
            b = bank["z"][idxs[1]].astype(np.float32)
            eb = bank["ev"][idxs[1]].astype(np.float32)
            d = b - a
            ed = eb - ea
        prompt = build_prompt(s, anchors[idxs[0]], anchors[idxs[1]] if two else "", two)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids], dtype=torch.long, device=device)
        emb = model.get_input_embeddings()(ids).float()
        sa = projector(torch.from_numpy(np.concatenate([a, ea]))[None].to(device))
        sb = projector(torch.from_numpy(np.concatenate([b, eb]))[None].to(device))
        sd = projector(torch.from_numpy(np.concatenate([d, ed]))[None].to(device))
        for name, soft in [("<ECG_A>", sa), ("<ECG_B>", sb), ("<ECG_D>", sd)]:
            pos = (ids[0] == tok_ids[name]).nonzero(as_tuple=False).flatten()
            if len(pos):
                emb[0, pos] = soft[0, :len(pos)].to(emb.dtype)
        emb = emb.half()
        with torch.no_grad():
            gen = model.generate(
                inputs_embeds=emb,
                attention_mask=torch.ones(1, emb.size(1), device=device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0], skip_special_tokens=True).strip()
        pred = norm(text)
        gold = [norm(x) for x in (s["answer"] if isinstance(s["answer"], list) else [str(s["answer"])])]
        ok = any(pred == g or g in pred for g in gold)
        em, f1, pred_set, gold_set = set_scores(text, s["answer"], cmap.get(s["template_id"], []))
        legacy_correct += int(ok)
        set_em_sum += em
        set_f1_sum += f1
        qt = s["question_type"]
        rec = by_type.setdefault(qt, {"n": 0, "correct": 0, "set_em": 0.0, "set_f1": 0.0})
        rec["n"] += 1
        rec["correct"] += int(ok)
        rec["set_em"] += em
        rec["set_f1"] += f1
        if len(out) < 80:
            out.append({"i": k, "question": s["question"], "gold": s["answer"], "pred": text,
                        "correct": ok, "set_em": em, "set_f1": f1,
                        "pred_set": pred_set, "gold_set": gold_set, "type": qt})
        if (k + 1) % 50 == 0:
            print(k + 1, legacy_correct / (k + 1), set_em_sum / (k + 1), set_f1_sum / (k + 1), flush=True)
    by_type_out = {}
    for qt, v in by_type.items():
        by_type_out[qt] = {
            "n": v["n"],
            "legacy_correct": v["correct"],
            "legacy_acc": v["correct"] / max(1, v["n"]),
            "set_em": v["set_em"] / max(1, v["n"]),
            "set_f1": v["set_f1"] / max(1, v["n"]),
        }
    res = {
        "n": len(items),
        "legacy_accuracy": legacy_correct / max(1, len(items)),
        "set_em": set_em_sum / max(1, len(items)),
        "set_f1": set_f1_sum / max(1, len(items)),
        "by_type": by_type_out,
        "examples": out,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
