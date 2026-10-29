"""检索质量评估。

输出四种检索方案的对比：
    vector_only      只用向量召回
    bm25_only        只用 BM25 召回
    hybrid_weighted  向量 + BM25 归一化加权
    hybrid_rrf       向量 + BM25 排名融合

指标：Recall@1 / Recall@3 / Recall@5 / MRR。
判定方式：检索到的块文本中包含该条问题的任一 gold 关键词即算命中。

用法：
    python -m scripts.evaluate
    python -m scripts.evaluate --sweep     # 额外扫描 alpha 取值
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_DIR, HYBRID_ALPHA  # noqa: E402
from src.index import HybridRetriever  # noqa: E402

KS = (1, 3, 5)


def load_qa(path: Path) -> tuple[list[dict], list[dict]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [i for i in items if i.get("type") != "unanswerable" and i.get("keywords")]
    unanswerable = [i for i in items if i.get("type") == "unanswerable"]
    return answerable, unanswerable


def is_hit(hit_text: str, keywords: list[str]) -> bool:
    return any(kw in hit_text for kw in keywords)


def evaluate_config(
    retriever: HybridRetriever,
    qa: list[dict],
    mode: str,
    fusion: str = "weighted",
    alpha: float = HYBRID_ALPHA,
    top_k: int = 10,
) -> dict:
    recall = {k: 0 for k in KS}
    mrr_sum = 0.0
    misses: list[str] = []

    for item in qa:
        hits = retriever.search(item["q"], top_k=top_k, mode=mode, fusion=fusion, alpha=alpha)
        first_rank = None
        for rank, h in enumerate(hits, start=1):
            if is_hit(h.text, item["keywords"]):
                first_rank = rank
                break

        for k in KS:
            if first_rank is not None and first_rank <= k:
                recall[k] += 1
        mrr_sum += 1.0 / first_rank if first_rank else 0.0

        if first_rank is None or first_rank > 5:
            misses.append(item["q"])

    n = len(qa)
    return {
        "name": f"{mode}" + (f"/{fusion}/α={alpha:.2f}" if mode == "hybrid" else ""),
        "recall@1": recall[1] / n,
        "recall@3": recall[3] / n,
        "recall@5": recall[5] / n,
        "mrr": mrr_sum / n,
        "misses": misses,
    }


def fmt_row(r: dict) -> str:
    return (
        f"| {r['name']:<26} | {r['recall@1']:.3f} | {r['recall@3']:.3f} "
        f"| {r['recall@5']:.3f} | {r['mrr']:.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=EVAL_DIR / "qa_set.jsonl")
    parser.add_argument("--sweep", action="store_true", help="扫描混合权重 alpha")
    args = parser.parse_args()

    qa, unanswerable = load_qa(args.qa)
    print(f"评估集：{len(qa)} 条可回答问题，{len(unanswerable)} 条应拒答问题\n")

    retriever = HybridRetriever.load()

    results = [
        evaluate_config(retriever, qa, mode="vector"),
        evaluate_config(retriever, qa, mode="bm25"),
        evaluate_config(retriever, qa, mode="hybrid", fusion="weighted", alpha=HYBRID_ALPHA),
        evaluate_config(retriever, qa, mode="hybrid", fusion="rrf", alpha=HYBRID_ALPHA),
    ]

    header = "| 方案 | Recall@1 | Recall@3 | Recall@5 | MRR |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [header, sep] + [fmt_row(r) for r in results]
    table = "\n".join(lines)
    print(table)

    for r in results:
        if r["misses"]:
            print(f"\n[{r['name']}] 未在前 5 命中 {len(r['misses'])} 条：")
            for q in r["misses"][:5]:
                print(f"  - {q}")

    if args.sweep:
        print("\n=== alpha 权重扫描（hybrid/weighted）===")
        sweep = []
        for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0):
            sweep.append(evaluate_config(retriever, qa, mode="hybrid", fusion="weighted", alpha=a))
        print("\n".join([header, sep] + [fmt_row(r) for r in sweep]))
        best = max(sweep, key=lambda r: (r["recall@5"], r["mrr"]))
        print(f"\n最佳 alpha：{best['name']}，MRR={best['mrr']:.3f}")

    report = (
        "# 检索评估报告\n\n"
        f"- 评估集：{len(qa)} 条可回答问题 + {len(unanswerable)} 条应拒答问题\n"
        f"- 判定口径：前 k 条检索结果中包含 gold 关键词即算命中\n"
        f"- 指标：Recall@1/3/5 与 MRR（Mean Reciprocal Rank）\n\n"
        f"{table}\n"
    )
    (EVAL_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {EVAL_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
