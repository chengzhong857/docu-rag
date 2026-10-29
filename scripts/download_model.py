"""从 ModelScope 下载向量模型到本地。

为什么需要这个脚本：huggingface.co 与 hf-mirror.com 在国内网络下经常不可达，
ModelScope（魔搭）是稳定的替代源。下载到本地后，sentence-transformers
可以直接加载本地目录，不再依赖任何外网。

用法：
    python -m scripts.download_model
    python -m scripts.download_model --repo AI-ModelScope/bge-small-zh-v1.5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent

# sentence-transformers 加载一个模型所必需的文件
REQUIRED_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docu-rag/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                tmp = dest.with_suffix(dest.suffix + ".part")
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done / total * 100
                            print(f"\r    {dest.name}: {pct:5.1f}% ({done // 1024} KB)", end="", flush=True)
                print()
                tmp.replace(dest)
            return
        except Exception as exc:
            print(f"\n    [重试 {attempt}/{retries}] {dest.name}: {exc}")
            if attempt == retries:
                raise
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="AI-ModelScope/bge-small-zh-v1.5")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args()

    target = args.target or BASE_DIR / "models" / args.repo.split("/")[-1]
    print(f"从 ModelScope 下载 {args.repo}\n目标目录：{target}\n")

    for name in REQUIRED_FILES:
        url = API.format(repo=args.repo, path=urllib.parse.quote(name, safe=""))
        dest = target / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  跳过（已存在）{name}")
            continue
        print(f"  下载 {name}")
        fetch(url, dest)

    # 清理 sentence-transformers 不认识的冗余文件
    extra = target / "pytorch_model.bin"
    if extra.exists():
        extra.unlink()

    print(f"\n完成，模型路径：{target}")
    print("在 .env 中设置：")
    print(f"EMBED_MODEL={target.as_posix()}")


if __name__ == "__main__":
    main()
