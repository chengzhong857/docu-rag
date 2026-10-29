"""命令行建库：解析 -> 切分 -> 向量化 -> 写入索引。

用法：
    python -m scripts.ingest
    python -m scripts.ingest --docs data/docs --reset
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 允许以脚本方式直接运行（python scripts/ingest.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_DIR  # noqa: E402
from src.index import HybridRetriever  # noqa: E402
from src.loader import load_directory  # noqa: E402
from src.splitter import split_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建文档索引")
    parser.add_argument("--docs", type=Path, default=DOCS_DIR, help="文档目录")
    parser.add_argument("--reset", action="store_true", help="清空旧索引后重建")
    args = parser.parse_args()

    if not args.docs.exists():
        raise SystemExit(f"文档目录不存在：{args.docs}")

    t0 = time.perf_counter()
    docs = load_directory(args.docs)
    if not docs:
        raise SystemExit(f"目录下没有可解析的文档：{args.docs}")

    total_chars = sum(d.char_count for d in docs)
    print(f"解析完成：{len(docs)} 个文件，共 {total_chars} 字")
    for d in docs:
        print(f"  - {d.source}：{d.char_count} 字 / {len(d.pages)} 页")

    chunks = split_documents(docs)
    lengths = [len(c.text) for c in chunks]
    print(
        f"切分完成：{len(chunks)} 块，"
        f"平均 {sum(lengths) / len(lengths):.0f} 字，"
        f"最短 {min(lengths)} / 最长 {max(lengths)} 字"
        f"（目标 {CHUNK_SIZE}，重叠 {CHUNK_OVERLAP}）"
    )

    if args.reset and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        print("已清空旧索引")

    retriever = HybridRetriever.build([c.to_dict() for c in chunks], show_progress=True)
    retriever.save(INDEX_DIR)

    elapsed = time.perf_counter() - t0
    print(f"索引已写入 {INDEX_DIR}，耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
