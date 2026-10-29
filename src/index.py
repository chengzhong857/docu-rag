"""混合检索层：FAISS 向量检索 + BM25 关键词检索。

为什么要做混合检索（这是本项目最值得讲的设计点）：
- 纯向量检索对「语义相近但字面不同」的查询强，但对专有名词、编号、缩写、
  精确术语常常失灵。比如问「BGE 模型的向量维度是多少」，「512」这种数字
  在向量空间里几乎不携带语义，全靠关键词命中的 BM25 才能召回来。
- 纯 BM25 反过来，问「怎么样防止模型背题」它抓不到「过拟合」这个词。
两者互补，所以融合。

融合提供两种方案：
1. weighted：各自 min-max 归一化后加权求和，权重可调（HYBRID_ALPHA）；
2. rrf：Reciprocal Rank Fusion，只看排名不看分数，对两路分数量纲不一致更鲁棒。
评估脚本会对这四种组合做对比。

关于 BM25 变体的踩坑记录（真实测出来的，不是抄文档）：
一开始试过 rank_bm25 的 BM25Plus，理由是它不会出现负 IDF。实测发现它在
小语料 + 短查询下会退化：它的评分含一个与词频无关的 delta 项，导致
**任何文档只要查询词在全局有 IDF 就会拿到基础分**，实测 44 个块全部得分非零，
最低分恰好等于查询词 IDF 之和。当查询里的词在语料中命中很少时，
大量零重叠的块会挤进 top-k，排序完全失效。
BM25Okapi 没有这个无条件加分项，零重叠文档精确得 0 分，于是换回来了。
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .config import (
    BM25_PATH,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    HYBRID_ALPHA,
    INDEX_DIR,
    RERANK_MODEL,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
    USE_RERANK,
)
from .embedder import encode_query, encode_texts

# 用搜索引擎模式切词，长词会额外切出子词，提升中文专有名词的召回率
jieba.setLogLevel(20)

# 拉丁字母开头的标识符整体保留：BatchNorm、bge-small-zh-v1.5、pytorch_model.bin
# 直接丢给 jieba 会被切成 batch/norm/layer/norm 这类碎片，关键词信号就消失了
_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.+#/-]*")
# 中文部分的停用词：疑问词、语气词、高频虚词。它们几乎不携带检索信息，
# 却会在小语料上因为文档频率高而扰乱 IDF 权重。
_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "为", "对", "从", "把", "被",
    "这", "那", "之", "其", "并", "且", "但", "而", "也", "就", "都", "还", "又",
    "很", "更", "最", "有", "会", "能", "要", "我", "你", "他", "她",
    # 注意：「不」「没」不能进停用词表，去掉会把「不支持删除」变成「支持删除」，
    # 语义直接反转。否定词必须保留。
    "它", "们", "个", "些", "上", "下", "中", "里", "时", "后", "前",
    "什么", "怎么", "怎样", "如何", "多少", "哪些", "哪个", "为什么",
    "应该", "可以", "一般", "通常", "比较", "一个", "这个", "那个", "以及",
}


def tokenize(text: str) -> list[str]:
    """中英混合分词。

    分两步：先把拉丁字母/数字开头的串整体抽出并小写化，再把剩下的中文交给 jieba。
    这样「BatchNorm」是一个完整词，而不是 batch + norm 两个碎片。
    """
    tokens = [m.group(0).lower() for m in _ASCII_TOKEN.finditer(text)]
    zh_only = _ASCII_TOKEN.sub(" ", text)
    for tok in jieba.cut_for_search(zh_only):
        tok = tok.strip()
        if not tok or tok in _STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isalnum():  # 丢掉单个标点
            continue
        tokens.append(tok)
    return tokens


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    page_no: int
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page_no": self.page_no,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "fused_score": round(self.fused_score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:  # 全部相等时不做缩放，避免除零
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, chunks: list[dict], index: faiss.Index, bm25: BM25Okapi):
        self.chunks = chunks
        self.index = index
        self.bm25 = bm25
        self._reranker = None

    # ---------------- 构建 ----------------
    @classmethod
    def build(cls, chunks: list[dict], show_progress: bool = True) -> "HybridRetriever":
        texts = [c["text"] for c in chunks]
        vecs = encode_texts(texts, show_progress=show_progress)

        # 向量已 L2 归一化，内积即余弦相似度，用 IndexFlatIP 精确检索
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        bm25 = BM25Okapi([tokenize(t) for t in texts])
        return cls(chunks=chunks, index=index, bm25=bm25)

    # ---------------- 检索 ----------------
    def _vector_search(self, query: str, k: int) -> dict[int, float]:
        q = encode_query(query).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        return {int(i): float(s) for s, i in zip(scores[0], idxs[0]) if i >= 0}

    def _bm25_search(self, query: str, k: int) -> dict[int, float]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][: min(k, len(scores))]
        return {int(i): float(scores[i]) for i in order}

    @staticmethod
    def _fuse(
        vec: dict[int, float],
        bm: dict[int, float],
        alpha: float,
        fusion: str,
    ) -> dict[int, float]:
        candidates = set(vec) | set(bm)

        if fusion == "rrf":
            # 只看排名：score = Σ 1/(k + rank)，k 取 60 是 RRF 的经验默认值
            fused: dict[int, float] = {}
            k_const = 60
            vec_rank = {i: r for r, i in enumerate(sorted(vec, key=vec.get, reverse=True), 1)}
            bm_rank = {i: r for r, i in enumerate(sorted(bm, key=bm.get, reverse=True), 1)}
            for i in candidates:
                s = 0.0
                if i in vec_rank:
                    s += alpha / (k_const + vec_rank[i])
                if i in bm_rank:
                    s += (1 - alpha) / (k_const + bm_rank[i])
                fused[i] = s
            return fused

        # weighted：先归一化到同一量纲再加权
        idxs = sorted(candidates)
        v_arr = np.array([vec.get(i, 0.0) for i in idxs])
        b_arr = np.array([bm.get(i, 0.0) for i in idxs])
        v_norm, b_norm = _minmax(v_arr), _minmax(b_arr)
        return {i: alpha * v + (1 - alpha) * b for i, v, b in zip(idxs, v_norm, b_norm)}

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            print(f"[retriever] 加载重排模型：{RERANK_MODEL}")
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        alpha: float = HYBRID_ALPHA,
        fusion: str = "weighted",
        mode: str = "hybrid",  # hybrid | vector | bm25
        rerank: bool | None = None,
    ) -> list[Hit]:
        use_rerank = USE_RERANK if rerank is None else rerank

        if mode == "vector":
            vec = self._vector_search(query, max(TOP_K_VECTOR, top_k))
            fused = vec
            bm: dict[int, float] = {}
        elif mode == "bm25":
            bm = self._bm25_search(query, max(TOP_K_BM25, top_k))
            vec = {}
            fused = bm
        else:
            vec = self._vector_search(query, TOP_K_VECTOR)
            bm = self._bm25_search(query, TOP_K_BM25)
            fused = self._fuse(vec, bm, alpha, fusion)

        # 重排需要更大的候选池，否则重排没有意义
        pool = max(top_k, TOP_K_VECTOR if use_rerank else top_k)
        top_idx = sorted(fused, key=fused.get, reverse=True)[:pool]

        hits = [
            Hit(
                chunk_id=self.chunks[i]["chunk_id"],
                text=self.chunks[i]["text"],
                source=self.chunks[i]["source"],
                page_no=self.chunks[i]["page_no"],
                vector_score=vec.get(i, 0.0),
                bm25_score=bm.get(i, 0.0),
                fused_score=fused[i],
            )
            for i in top_idx
        ]

        if use_rerank and hits:
            reranker = self._get_reranker()
            pairs = [(query, h.text) for h in hits]
            scores = reranker.predict(pairs)
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)

        return hits[:top_k]

    # ---------------- 持久化 ----------------
    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / FAISS_INDEX_PATH.name))
        (directory / CHUNKS_PATH.name).write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(directory / BM25_PATH.name, "wb") as f:
            pickle.dump({"tokens": [c["text"] for c in self.chunks]}, f)

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "HybridRetriever":
        chunks_path = directory / CHUNKS_PATH.name
        if not chunks_path.exists():
            raise FileNotFoundError("索引不存在，请先运行 scripts/ingest.py 建库")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / FAISS_INDEX_PATH.name))
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        return cls(chunks=chunks, index=index, bm25=bm25)
