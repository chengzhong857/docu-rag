"""Streamlit 界面。

启动：
    streamlit run app.py

设计上刻意把「检索到的原文 + 各路得分」全部暴露出来，而不是只显示最终答案。
原因是 RAG 出问题时九成出在检索，而不是生成，使用者需要能一眼看出
是没召回到、还是召回到了但模型没用上。
"""
from __future__ import annotations

import streamlit as st

from src.config import HYBRID_ALPHA, TOP_K_FINAL
from src.generator import is_llm_available
from src.index import HybridRetriever
from src.pipeline import RagPipeline

st.set_page_config(page_title="文档问答 · RAG Demo", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="正在加载索引与向量模型…")
def get_pipeline() -> RagPipeline:
    return RagPipeline(HybridRetriever.load())


def render_hits(hits) -> None:
    for i, h in enumerate(hits, start=1):
        score_bits = [f"融合 {h.fused_score:.3f}", f"向量 {h.vector_score:.3f}", f"BM25 {h.bm25_score:.2f}"]
        if h.rerank_score is not None:
            score_bits.append(f"重排 {h.rerank_score:.3f}")
        with st.expander(f"[{i}] {h.source} · 第 {h.page_no} 页 · " + " | ".join(score_bits)):
            st.markdown(h.text)


def main() -> None:
    st.title("📄 文档问答 · RAG Demo")
    st.caption("混合检索（向量 + BM25）+ 引用溯源 · 支持 PDF / DOCX / TXT / Markdown")

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m scripts.ingest", language="bash")
        return

    with st.sidebar:
        st.subheader("检索参数")
        mode_label = st.selectbox(
            "检索模式",
            ["混合检索", "仅向量", "仅 BM25"],
            help="用来对比不同召回策略的效果差异",
        )
        fusion = st.radio("融合方式", ["weighted", "rrf"], horizontal=True,
                          help="weighted 按归一化分数加权；rrf 只看排名")
        alpha = st.slider("向量权重 α", 0.0, 1.0, HYBRID_ALPHA, 0.05,
                          help="α 越大越依赖语义，越小越依赖关键词")
        top_k = st.slider("送入上下文的片段数", 1, 10, TOP_K_FINAL)

        st.divider()
        st.caption(f"知识库规模：{len(pipeline.retriever.chunks)} 个文本块")
        if is_llm_available():
            st.success("大模型已配置")
        else:
            st.warning("未配置 API Key，当前为检索模式\n\n在 .env 中填入 LLM_API_KEY 即可启用生成")

    mode = {"混合检索": "hybrid", "仅向量": "vector_only", "仅 BM25": "bm25_only"}[mode_label]

    question = st.chat_input("针对文档内容提问，例如：中文文档做 RAG 时每块切多少字合适？")
    if not question:
        st.info("提示：知识库当前包含机器学习、深度学习、Transformer、RAG、向量检索、评估指标六个主题的文档。")
        return

    with st.spinner("检索中…"):
        result = pipeline.ask(question, top_k=top_k, alpha=alpha, mode=mode, fusion=fusion)

    st.markdown(f"**问题**　{question}")
    st.markdown("### 回答")
    st.markdown(result.answer)

    st.markdown(f"### 引用来源（{len(result.hits)} 条）")
    if result.degraded:
        st.caption("检索模式下直接展示原文片段；配置 API Key 后由大模型归纳并标注引用编号。")
    render_hits(result.hits)

    st.caption(
        f"模式 {result.mode} · 融合 {fusion} · α={alpha:.2f} · "
        f"耗时 {result.latency_ms} ms"
    )


if __name__ == "__main__":
    main()
