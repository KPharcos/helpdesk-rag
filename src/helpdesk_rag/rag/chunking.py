"""Heading-aware markdown chunker.

Each document is split at markdown headings first (so a chunk never mixes two topics), then
long sections are split into windows of at most `chunk_size` characters with `overlap`
characters carried over, cutting at paragraph / sentence / word boundaries where possible.
"""

from __future__ import annotations

import re
from pathlib import Path

from helpdesk_rag.domain import Chunk

# A closing run of `#` only counts after whitespace (CommonMark), so "## Learning C#" keeps its `#`.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of <= size chars with ~overlap chars shared between neighbours."""
    if size <= 0 or not 0 <= overlap < size // 2:
        raise ValueError("require size > 0 and 0 <= overlap < size // 2")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer the latest natural boundary in the second half of the window.
            floor = start + size // 2
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        # Next window starts `overlap` chars back, snapped forward to a word start.
        next_start = end - overlap
        if overlap:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = max(next_start, start + 1)
    return pieces


def _sections(markdown: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Return (title, [(section_heading, body), ...]). Title = first level-1 heading."""
    title: str | None = None
    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))
        buffer.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1 and title is None:
                flush()
                title = heading
                current_heading = ""
                continue
            flush()
            current_heading = heading
            continue
        buffer.append(line)
    flush()
    return title, sections


def chunk_markdown(doc_id: str, markdown: str, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    title, sections = _sections(markdown)
    title = title or doc_id.replace("-", " ").replace("_", " ").title()
    chunks: list[Chunk] = []
    for heading, body in sections:
        section = heading or "Overview"
        for piece in split_text(body, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def load_and_chunk(docs_dir: Path, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No *.md documents found in {docs_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(
            chunk_markdown(
                path.stem,
                path.read_text(encoding="utf-8-sig"),  # tolerate a BOM from Windows editors
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
