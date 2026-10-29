"""中文感知的文本切分。

为什么不直接用 LangChain 的 RecursiveCharacterTextSplitter：
它默认的分隔符优先级对中文不友好，遇到「。」之外的断句（分号、冒号、列表项）
容易把一句完整的话切成两半，导致 embedding 语义被打散。

这里的策略：
1. 先按 Markdown 标题 / 空行做「结构切分」，标题作为新块的起点（保留层级语义）；
2. 再按中文标点切句，累积到接近 CHUNK_SIZE 时落块；
3. 相邻块保留 CHUNK_OVERLAP 字的重叠，避免答案正好跨在两块边界上而丢失；
4. 超长句（无标点的长段落）兜底做滑动窗口硬切。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .loader import ParsedDoc

# 中文句子结束符 + 英文句末 + 换行
_SENT_END = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s+|\n+")
# Markdown 标题行
_HEADING = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_no: int
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text)]
    return [p for p in parts if p]


def _hard_split(sentence: str, size: int, overlap: int) -> list[str]:
    """没有标点的超长段落，按固定窗口滑动切。"""
    if len(sentence) <= size:
        return [sentence]
    step = max(1, size - overlap)
    return [sentence[i : i + size] for i in range(0, len(sentence), step) if sentence[i : i + size]]


def _blocks_with_headings(text: str) -> list[str]:
    """按标题把文档切成若干结构块，标题留在块首以便检索时带上语境。"""
    positions = [m.start() for m in _HEADING.finditer(text)]
    if not positions:
        return [text]

    blocks: list[str] = []
    if positions[0] > 0:
        head = text[: positions[0]].strip()
        if head:
            blocks.append(head)
    for idx, start in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _split_block(block: str, size: int, overlap: int) -> list[str]:
    sentences: list[str] = []
    for sent in _split_sentences(block):
        sentences.extend(_hard_split(sent, size, overlap))

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        if buf and buf_len + len(sent) > size:
            chunks.append("".join(buf).strip())
            # 构造重叠：从缓冲区尾部回取，直到达到 overlap 字数
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(buf):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            buf, buf_len = tail, tail_len
        buf.append(sent)
        buf_len += len(sent)

    if buf:
        chunks.append("".join(buf).strip())
    return [c for c in chunks if c]


def split_document(doc: ParsedDoc, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """把一篇文档切成带元数据的块。"""
    chunks: list[Chunk] = []
    idx = 0
    for page in doc.pages:
        for block in _blocks_with_headings(page.text):
            for piece in _split_block(block, size, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.source}#p{page.page_no}#c{idx}",
                        text=piece,
                        source=doc.source,
                        page_no=page.page_no,
                        chunk_index=idx,
                    )
                )
                idx += 1
    return chunks


def split_documents(docs: list[ParsedDoc]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(split_document(doc))
    return out
