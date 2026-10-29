#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

SPECIAL = ["<ECG_A>", "<ECG_B>", "<ECG_D>"]


class QADataset(Dataset):
    def __init__(self, jsonl, bank, anchors, id2idx, tokenizer, max_len=1024, train=True, limit=0, drop_anchor=False, zero_image=False, signal_only=False):
        self.items = [json.loads(x) for x in open(jsonl, encoding="utf-8")]
        if limit:
            self.items = self.items[:limit]
        self.z = bank["z"]
        self.ev = bank["ev"]
        self.anchors = anchors
        self.id2idx = id2idx
        self.tok = tokenizer
        self.max_len = max_len
        self.train = train
        self.drop_anchor = drop_anchor or signal_only
        self.zero_image = zero_image
        self.signal_only = signal_only

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        ids = s["ecg_ids"]
        idxs = [self.id2idx[str(x)] for x in ids]
        a = self.z[idxs[0]].astype(np.float32)
        ea = self.ev[idxs[0]].astype(np.float32)
        anchor_a = self.anchors[idxs[0]]
        b = np.zeros_like(a); eb = np.zeros_like(ea); d = np.zeros_like(a); ed = np.zeros_like(ea)
        anchor_b = ""
        if len(idxs) > 1:
            b = self.z[idxs[1]].astype(np.float32)
            eb = self.ev[idxs[1]].astype(np.float32)
            anchor_b = self.anchors[idxs[1]]
            d = b - a; ed = eb - ea
        if self.zero_image:
            a[512:] = 0; b[512:] = 0; d[512:] = 0
        if self.signal_only:
            soft_a = a[:512]; soft_b = b[:512]; soft_d = d[:512]
        else:
            soft_a = np.concatenate([a, ea]); soft_b = np.concatenate([b, eb]); soft_d = np.concatenate([d, ed])
        ans = "; ".join(s["answer"]) if isinstance(s["answer"], list) else str(s["answer"])
        if len(idxs) > 1:
            prompt = (
                "<|im_start|>system\nYou are an ECG interpretation assistant. Use the evidence anchors and delta evidence.\n<|im_end|>\n"
                "<|im_start|>user\n"
                "ECG_A: " + "<ECG_A>" * 8 + "\n"
                "ECG_B: " + "<ECG_B>" * 8 + "\n" +
                "Delta: " + "<ECG_D>" * 8 + "\n" +
                ("" if self.drop_anchor else f"Evidence A: {anchor_a}\nEvidence B: {anchor_b}\n") +
                f"Question: {s['question']}\n<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            prompt = (
                "<|im_start|>system\nYou are an ECG interpretation assistant. Use the evidence anchor.\n<|im_end|>\n"
                "<|im_start|>user\n"
                "ECG: " + "<ECG_A>" * 8 + "\n" +
                ("" if self.drop_anchor else f"Evidence: {anchor_a}\n") +
                f"Question: {s['question']}\n<|im_end|>\n<|im_start|>assistant\n"
            )
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(ans + "<|im_end|>", add_special_tokens=False)["input_ids"]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        if len(input_ids) > self.max_len:
            if len(p_ids) < self.max_len:
                keep = self.max_len - len(p_ids)
                input_ids = p_ids + a_ids[:keep]
                labels = [-100] * len(p_ids) + a_ids[:keep]
            else:
                input_ids = p_ids[:self.max_len - 1] + a_ids[:1]
                labels = [-100] * (self.max_len - 1) + a_ids[:1]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "soft_a": torch.from_numpy(soft_a), "soft_b": torch.from_numpy(soft_b), "soft_d": torch.from_numpy(soft_d),
            "has_b": torch.tensor(1 if len(idxs) > 1 else 0, dtype=torch.long),
            "meta": s,
        }


def collate(batch, pad_id):
    L = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = x["input_ids"]; labels[i, :n] = x["labels"]; attn[i, :n] = 1
    return {
        "input_ids": input_ids, "labels": labels, "attention_mask": attn,
        "soft_a": torch.stack([x["soft_a"] for x in batch]),
        "soft_b": torch.stack([x["soft_b"] for x in batch]),
        "soft_d": torch.stack([x["soft_d"] for x in batch]),
        "has_b": torch.stack([x["has_b"] for x in batch]),
        "meta": [x["meta"] for x in batch],
    }


class Projector(nn.Module):
    def __init__(self, in_dim, hidden, n_tokens, out_dim):
        super().__init__()
        self.n = n_tokens; self.out = out_dim
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n_tokens * out_dim))

    def forward(self, x):
        return self.net(x).view(x.size(0), self.n, self.out)


def build_embeds(model, batch, projector, tok_ids, device):
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(batch["input_ids"].to(device)).float()
    sa = projector(batch["soft_a"].to(device))
    sb = projector(batch["soft_b"].to(device))
    sd = projector(batch["soft_d"].to(device))
    ids = batch["input_ids"].to(device)
    for b in range(ids.size(0)):
        for name, soft in [("<ECG_A>", sa), ("<ECG_B>", sb), ("<ECG_D>", sd)]:
            pos = (ids[b] == tok_ids[name]).nonzero(as_tuple=False).flatten()
            if len(pos):
                emb[b, pos] = soft[b, :len(pos)].to(emb.dtype)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train-jsonl", default="")
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--init-projector", default="")
    ap.add_argument("--drop-anchor", action="store_true")
    ap.add_argument("--zero-image", action="store_true")
    ap.add_argument("--signal-only", action="store_true")
    ap.add_argument("--eval-samples", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-soft", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    tok_ids = {s: tok.convert_tokens_to_ids(s) for s in SPECIAL}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    bank = np.load(args.bank)
    anchors = json.load(open(os.path.join(args.qa_dir, "anchors.json"), encoding="utf-8"))
    id2idx = json.load(open(os.path.join(args.qa_dir, "id2idx.json"), encoding="utf-8"))
    proj_dim = 512 if args.signal_only else 1052
    projector = Projector(proj_dim, 1024, args.n_soft, 2048).to(device)
    if args.init_projector:
        projector.load_state_dict(torch.load(args.init_projector, map_location=device))
    train_path = args.train_jsonl or os.path.join(args.qa_dir, "train.jsonl")
    ds = QADataset(train_path, bank, anchors, id2idx, tok, args.max_len, True, args.max_samples, args.drop_anchor, args.zero_image, args.signal_only)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=0)
    opt = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train(); projector.train()
        for i, batch in enumerate(dl):
            emb = build_embeds(model, batch, projector, tok_ids, device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(inputs_embeds=emb, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
                if step % 20 == 0:
                    print(f"step {step} loss {float(loss.detach())*args.grad_accum:.4f} elapsed {time.time()-t0:.0f}s", flush=True)
                if args.save_steps and step % args.save_steps == 0:
                    model.save_pretrained(os.path.join(args.out_dir, f"adapter_step{step}"))
                    torch.save(projector.state_dict(), os.path.join(args.out_dir, f"projector_step{step}.pt"))
        if args.max_steps and step >= args.max_steps:
            break
    model.save_pretrained(os.path.join(args.out_dir, "adapter"))
    torch.save(projector.state_dict(), os.path.join(args.out_dir, "projector.pt"))
    print("training done", args.out_dir)


if __name__ == "__main__":
    main()







