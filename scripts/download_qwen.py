#!/usr/bin/env python3
import os
import sys

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from huggingface_hub import snapshot_download

dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained", "Qwen2.5-3B-Instruct")
os.makedirs(dst, exist_ok=True)
snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir=dst,
    ignore_patterns=["*.bin", "*.pth", "*.pt", "*.gguf"],
)
print("done", dst)
