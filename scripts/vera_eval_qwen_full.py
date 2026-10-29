#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from vera_train_qwen import Projector, SPECIAL
from vera_eval_qwen import build_prompt, norm, set_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--answers-csv", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--question-type", default="")
    ap.add_argument("--ecg-id", type=int, default=0)
    ap.add_argument("--question-contains", default="")
    ap.add_argument("--drop-anchor", action="store_true")
    ap.add_argument("--drop-image", action="store_true")
    ap.add_argument("--drop-delta", action="store_true")
    ap.add_argument("--signal-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
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
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    proj_dim = 512 if args.signal_only else 1052
    projector = Projector(proj_dim, 1024, 8, 2048).to(device)
    projector.load_state_dict(torch.load(args.projector, map_location=device))
    projector.eval()
    emb_layer = model.get_input_embeddings()

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
            random.seed(0); random.shuffle(v)
            items.extend(v[:per])
    elif args.all:
        items = items_all
    else:
        items = items_all[:args.samples]
    if args.question_type:
        items = [x for x in items if x["question_type"] == args.question_type]
    if args.ecg_id:
        items = [x for x in items if args.ecg_id in x["ecg_ids"]]
    if args.question_contains:
        items = [x for x in items if args.question_contains.lower() in x["question"].lower()]

    out = []
    legacy_correct = 0
    set_em_sum = 0.0
    set_f1_sum = 0.0
    by_type = {}
    gen_times = []
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    for start in range(0, len(items), args.batch_size):
        batch_items = items[start:start + args.batch_size]
        p_list, soft_list, prompt_list = [], [], []
        for s in batch_items:
            idxs = [id2idx[str(x)] for x in s["ecg_ids"]]
            a = bank["z"][idxs[0]].astype(np.float32); ea = bank["ev"][idxs[0]].astype(np.float32)
            b = np.zeros_like(a); eb = np.zeros_like(ea); d = np.zeros_like(a); ed = np.zeros_like(ea)
            two = len(idxs) > 1
            if two:
                b = bank["z"][idxs[1]].astype(np.float32); eb = bank["ev"][idxs[1]].astype(np.float32)
                d = b - a; ed = eb - ea
            a0 = anchors[idxs[0]] if not (args.drop_anchor or args.signal_only) else ""
            b0 = anchors[idxs[1]] if two and not (args.drop_anchor or args.signal_only) else ""
            prompt = build_prompt(s, a0, b0, two)
            if args.drop_image:
                a[512:] = 0; b[512:] = 0; d[512:] = 0
            if args.drop_delta:
                d[:] = 0; ed[:] = 0
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            p_list.append(p_ids)
            prompt_list.append(prompt)
            if args.signal_only:
                fa, fb, fd = a[:512], b[:512], d[:512]
            else:
                fa, fb, fd = np.concatenate([a, ea]), np.concatenate([b, eb]), np.concatenate([d, ed])
            soft_list.append((
                torch.from_numpy(fa)[None].to(device),
                torch.from_numpy(fb)[None].to(device),
                torch.from_numpy(fd)[None].to(device),
            ))
        max_len = max(len(x) for x in p_list)
        ids = torch.full((len(p_list), max_len), tok.pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(p_list), max_len), dtype=torch.long, device=device)
        for i, pids in enumerate(p_list):
            n = len(pids)
            ids[i, max_len - n:] = torch.tensor(pids, dtype=torch.long, device=device)
            attn[i, max_len - n:] = 1
        emb = emb_layer(ids).float()
        for i, (sa_in, sb_in, sd_in) in enumerate(soft_list):
            sa = projector(sa_in); sb = projector(sb_in); sd = projector(sd_in)
            for name, soft in [("<ECG_A>", sa), ("<ECG_B>", sb), ("<ECG_D>", sd)]:
                pos = (ids[i] == tok_ids[name]).nonzero(as_tuple=False).flatten()
                if len(pos):
                    emb[i, pos] = soft[0, :len(pos)].to(emb.dtype)
        emb = emb.half()
        position_ids = (attn.cumsum(-1) - 1).clamp_min(0)
        t_gen = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                inputs_embeds=emb, attention_mask=attn, position_ids=position_ids,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                eos_token_id=eos, pad_token_id=tok.pad_token_id,
            )
        gen_times.append(time.perf_counter() - t_gen)
        for j, s in enumerate(batch_items):
            text = tok.decode(gen[j], skip_special_tokens=True).strip()
            pred = norm(text)
            gold = [norm(x) for x in (s["answer"] if isinstance(s["answer"], list) else [str(s["answer"])])]
            ok = any(pred == g or g in pred for g in gold)
            em, f1, pred_set, gold_set = set_scores(text, s["answer"], cmap.get(s["template_id"], []))
            legacy_correct += int(ok); set_em_sum += em; set_f1_sum += f1
            qt = s["question_type"]
            rec = by_type.setdefault(qt, {"n": 0, "correct": 0, "set_em": 0.0, "set_f1": 0.0})
            rec["n"] += 1; rec["correct"] += int(ok); rec["set_em"] += em; rec["set_f1"] += f1
            if len(out) < 80:
                out.append({"i": start + j, "question": s["question"], "gold": s["answer"], "pred": text,
                            "correct": ok, "set_em": em, "set_f1": f1,
                            "pred_set": pred_set, "gold_set": gold_set, "type": qt})
        done = start + len(batch_items)
        if (start // max(1, args.batch_size)) % 10 == 0:
            print(done, legacy_correct / done, set_em_sum / done, set_f1_sum / done, flush=True)

    by_type_out = {}
    for qt, v in by_type.items():
        by_type_out[qt] = {"n": v["n"], "legacy_correct": v["correct"],
                           "legacy_acc": v["correct"] / max(1, v["n"]),
                           "set_em": v["set_em"] / max(1, v["n"]),
                           "set_f1": v["set_f1"] / max(1, v["n"])}
    res = {"n": len(items), "legacy_accuracy": legacy_correct / max(1, len(items)),
           "latency_ms_per_sample": float(sum(gen_times) / max(1, len(items)) * 1000),
           "set_em": set_em_sum / max(1, len(items)), "set_f1": set_f1_sum / max(1, len(items)),
           "by_type": by_type_out, "examples": out}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()








