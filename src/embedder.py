"""文本向量化。

选型说明（面试常问）：为什么是 bge-small-zh-v1.5 而不是 OpenAI embedding 或 bge-large？

- 走本地推理，不产生 API 成本，也不依赖网络，个人项目和离线场景更可控；
- small 版模型约 95MB，512 维，CPU 上编码 1000 条短文本约 10~20 秒，
  对几万条以内的知识库完全够用；large 是 1024 维、约 1.3GB，CPU 上慢一个数量级，
  而 Recall 提升只有几个点，性价比不划算；
- 中文语义检索效果明显好于 multilingual-MiniLM 一类通用小模型。
"""
from __future__ import annotations

import os

import numpy as np

# 国内网络下 huggingface.co 常连不上，默认走镜像；可用 HF_ENDPOINT 覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关闭 tokenizers 的多进程告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer  # noqa: E402

from .config import BGE_QUERY_INSTRUCTION, EMBED_MODEL, USE_QUERY_INSTRUCTION  # noqa: E402

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """懒加载单例。模型加载要几秒，不要在每次查询时重复初始化。"""
    global _model
    if _model is None:
        print(f"[embedder] 加载向量模型：{EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _model


def encode_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """把文本编码成 L2 归一化后的向量。

    归一化之后，向量内积等价于余弦相似度，所以后面可以直接用 FAISS 的 IP 索引。
    """
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def encode_query(query: str) -> np.ndarray:
    """查询单独走一遍：BGE 训练时查询带指令前缀，加前缀能提升短查询的召回。"""
    text = f"{BGE_QUERY_INSTRUCTION}{query}" if USE_QUERY_INSTRUCTION else query
    return encode_texts([text])[0]
