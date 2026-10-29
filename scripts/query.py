"""命令行查询工具，用于快速验证检索效果、排查 badcase。

用法：
    python -m scripts.query "中文文档做 RAG 时每块切多少字比较合适？"
    python -m scripts.query "梯度消失怎么办" --top 3 --mode bm25
    python -m scripts.query "过拟合怎么看" --show-only-hit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import HybridRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--show-only-hit", action="store_true", help="只显示包含指定关键词的块")
    parser.add_argument("--keyword", default=None)
    args = parser.parse_args()

    retriever = HybridRetriever.load()
    hits = retriever.search(
        args.question, top_k=args.top, mode=args.mode, fusion=args.fusion, alpha=args.alpha
    )

    print(f"查询：{args.question}")
    print(f"模式：{args.mode} / {args.fusion} / α={args.alpha}\n")
    for rank, h in enumerate(hits, start=1):
        flag = ""
        if args.keyword and args.keyword in h.text:
            flag = "  <== 包含关键词"
        print(f"[{rank}] {h.chunk_id}  融合={h.fused_score:.4f} 向量={h.vector_score:.4f} BM25={h.bm25_score:.2f}{flag}")
        if not args.show_only_hit or flag:
            preview = h.text.replace("\n", " ")[:160]
            print(f"    {preview}…\n")


if __name__ == "__main__":
    main()
