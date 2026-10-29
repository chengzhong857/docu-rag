"""把「检索 → 生成」串成一条流水线。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import HYBRID_ALPHA, TOP_K_FINAL
from .generator import generate_answer, is_llm_available
from .index import Hit, HybridRetriever


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    mode: str = "hybrid"
    degraded: bool = False  # True 表示未调 LLM，只返回检索结果
    latency_ms: int = 0


class RagPipeline:
    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever.load()

    def ask(
        self,
        question: str,
        top_k: int = TOP_K_FINAL,
        alpha: float = HYBRID_ALPHA,
        mode: str = "hybrid",
        fusion: str = "weighted",
    ) -> Answer:
        t0 = time.perf_counter()
        if mode == "vector_only":
            hits = self.retriever.search(question, top_k=top_k, mode="vector")
        elif mode == "bm25_only":
            hits = self.retriever.search(question, top_k=top_k, mode="bm25")
        else:
            hits = self.retriever.search(question, top_k=top_k, alpha=alpha, fusion=fusion, mode="hybrid")

        degraded = not is_llm_available()
        if degraded:
            # 降级模式：把最相关的原文片段直接给用户，用于演示和检索效果调试
            text = "（未配置大模型 API Key，当前为检索模式，以下是与问题最相关的原文片段）"
        else:
            try:
                text = generate_answer(question, hits)
            except Exception as exc:
                degraded = True
                text = f"（生成失败：{exc}。以下为检索结果）"

        return Answer(
            question=question,
            answer=text,
            hits=hits,
            mode=mode,
            degraded=degraded,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
