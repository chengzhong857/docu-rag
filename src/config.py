"""全局配置。

所有可调参数集中在这里，方便做实验对比（改配置 → 重跑评估脚本 → 看指标变化）。
申请 API Key 后复制 .env.example 为 .env 并填入即可。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "index_store"
EVAL_DIR = BASE_DIR / "eval"

# ---------- 切分参数 ----------
# 中文一个汉字信息量约等于英文 1.5~2 个 token，420 字对应约 600~800 token，
# 是检索粒度与上下文完整性的折中值。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ---------- 检索参数 ----------


def _resolve_embed_model() -> str:
    """向量模型解析顺序：环境变量 > 本地 models 目录 > HuggingFace 仓库名。

    优先本地目录的原因：国内网络访问 huggingface.co 不稳定，把模型下到本地
    （scripts/download_model.py 可从 ModelScope 拉取）后加载不再依赖外网，
    启动速度也从十几秒降到两三秒。
    """
    env = os.getenv("EMBED_MODEL", "").strip()
    if env:
        return env
    local = BASE_DIR / "models" / "bge-small-zh-v1.5"
    if local.exists() and any(local.iterdir()):
        return local.as_posix()
    return "BAAI/bge-small-zh-v1.5"


EMBED_MODEL = _resolve_embed_model()
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "12"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "12"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
# 混合检索中向量得分的权重，0 为纯 BM25，1 为纯向量。
# 0.80 不是拍脑袋定的：scripts/evaluate.py --sweep 在本项目评估集上扫描
# 0.0/0.2/0.4/0.5/0.6/0.65/0.7/0.8/1.0 后，α=0.80 的 MRR 最高（0.975），
# 比初始经验值 0.65（0.942）高 3.3 个百分点。
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.80"))

# BGE 中文模型对「短查询检索长文档」场景建议加指令前缀，能提升召回
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
USE_QUERY_INSTRUCTION = os.getenv("USE_QUERY_INSTRUCTION", "1") == "1"

# 重排模型（可选，默认关闭：模型约 1.1GB，CPU 上会明显增加延迟）
USE_RERANK = os.getenv("USE_RERANK", "0") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# ---------- 生成参数 ----------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------- 存储路径 ----------
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25_corpus.pkl"
