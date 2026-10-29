"""文档解析层。

职责：把不同格式的文件统一解析成「带页码的文本段」列表。
页码必须保留，否则回答就无法溯源到原文位置，引用就成了摆设。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    """一页（或一个逻辑段落块）的纯文本。"""

    text: str
    page_no: int
    # 解析阶段的附加信息，比如 PDF 的页数、DOCX 的段落数
    extra: dict = field(default_factory=dict)


@dataclass
class ParsedDoc:
    source: str  # 文件名，用于引用展示
    pages: list[Page] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def _clean(text: str) -> str:
    """统一空白字符，去掉 PDF 解析常见的软换行与多余空格。"""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _load_pdf(path: Path) -> ParsedDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc = ParsedDoc(source=path.name)
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            doc.pages.append(Page(text=text, page_no=i))
    return doc


def _load_docx(path: Path) -> ParsedDoc:
    from docx import Document

    document = Document(str(path))
    paragraphs = [_clean(p.text) for p in document.paragraphs]
    paragraphs = [p for p in paragraphs if p]

    # DOCX 没有物理页概念。这里按约 1500 字切「逻辑页」，
    # 目的是让引用能定位到大致位置，而不是笼统地指向整个文件。
    doc = ParsedDoc(source=path.name)
    buf: list[str] = []
    buf_len = 0
    page_no = 1
    for para in paragraphs:
        buf.append(para)
        buf_len += len(para)
        if buf_len >= 1500:
            doc.pages.append(Page(text="\n".join(buf), page_no=page_no))
            buf, buf_len = [], 0
            page_no += 1
    if buf:
        doc.pages.append(Page(text="\n".join(buf), page_no=page_no))
    return doc


def _load_text(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean(raw)
    return ParsedDoc(source=path.name, pages=[Page(text=text, page_no=1)] if text else [])


def load_document(path: str | Path) -> ParsedDoc:
    """按扩展名分发解析器。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的格式：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    return _load_text(path)


def load_directory(directory: str | Path) -> list[ParsedDoc]:
    directory = Path(directory)
    docs: list[ParsedDoc] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                docs.append(load_document(path))
            except Exception as exc:  # 单个文件损坏不应中断整批入库
                print(f"[警告] 解析失败，已跳过 {path.name}：{exc}")
    return docs
