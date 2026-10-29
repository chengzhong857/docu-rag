"""回答生成层。

要点：
1. 走 OpenAI 兼容协议，所以 DeepSeek / 通义 / 智谱 / 本地 vLLM 都能直接用，
   只改 .env 里的 base_url 和 model 即可 —— 这是比「写死某一家 SDK」更好的工程习惯。
2. 没配 API Key 时自动降级为「仅检索」模式，项目依然能跑、能演示、能评估检索质量。
3. 提示词里强制要求编号引用，并允许模型在依据不足时明确拒答。
   拒答能力比强行编造更值钱 —— 这也是 RAG 相对纯 LLM 的核心价值。
"""
from __future__ import annotations

from openai import OpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from .index import Hit

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。请严格依据【参考资料】回答用户问题。\n"
    "规则：\n"
    "1. 只使用参考资料中的信息，不要引入外部知识，不要推测。\n"
    "2. 每个结论后面用方括号标注来源编号，例如 [1]、[2][3]。\n"
    "3. 如果参考资料不足以回答问题，直接回答「根据现有资料无法回答」，并说明缺什么信息。\n"
    "4. 回答使用简体中文，条理清晰，不要复述整段原文，要做归纳。"
)


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] 来源：{h.source} 第{h.page_no}页\n{h.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【参考资料】\n{build_context(hits)}\n\n【问题】\n{question}",
        },
    ]


def is_llm_available() -> bool:
    return bool(LLM_API_KEY)


def generate_answer(question: str, hits: list[Hit]) -> str:
    """调用大模型生成带引用的回答。未配置 Key 时抛错，由上层决定降级策略。"""
    if not is_llm_available():
        raise RuntimeError("未配置 LLM_API_KEY")

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(question, hits),
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content or ""
