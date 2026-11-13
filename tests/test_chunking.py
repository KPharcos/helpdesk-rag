from __future__ import annotations

import itertools

import pytest

from helpdesk_rag.rag.chunking import chunk_markdown, load_and_chunk, split_text
from tests.conftest import DOCS_DIR

DOC = """# VPN Guide

Intro paragraph about the VPN.

## Connecting

Open the client and click connect.

## Troubleshooting

### Certificate expired

Bring the laptop to the office.
"""


def test_sections_and_title() -> None:
    chunks = chunk_markdown("vpn", DOC, chunk_size=200, chunk_overlap=20)
    assert [c.section for c in chunks] == ["Overview", "Connecting", "Certificate expired"]
    assert {c.title for c in chunks} == {"VPN Guide"}
    assert [c.chunk_id for c in chunks] == ["vpn#0", "vpn#1", "vpn#2"]
    assert all(c.doc_id == "vpn" for c in chunks)
    # Empty "Troubleshooting" section (only a sub-heading) produces no chunk.
    assert "Troubleshooting" not in {c.section for c in chunks}


def test_title_falls_back_to_doc_id() -> None:
    chunks = chunk_markdown("printer-setup", "Just text.", chunk_size=200, chunk_overlap=20)
    assert chunks[0].title == "Printer Setup"
    assert chunks[0].section == "Overview"


def test_split_text_respects_size_and_overlaps() -> None:
    words = [f"word{i:03d}" for i in range(300)]
    text = " ".join(words)
    pieces = split_text(text, size=200, overlap=40)

    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    # Every word survives, in order.
    seen = " ".join(pieces).split()
    assert [w for i, w in enumerate(seen) if w not in seen[:i]] == words
    # Consecutive pieces share some words (overlap) and never cut a word in half.
    for left, right in itertools.pairwise(pieces):
        assert set(left.split()) & set(right.split())
        assert all(w in words for w in right.split())


def test_split_text_prefers_paragraph_boundaries() -> None:
    para1 = "First paragraph sentence. " * 5
    para2 = "Second paragraph sentence. " * 5
    pieces = split_text(f"{para1.strip()}\n\n{para2.strip()}", size=200, overlap=0)
    assert pieces[0] == para1.strip()


def test_split_text_short_and_empty() -> None:
    assert split_text("short", 100, 10) == ["short"]
    assert split_text("   ", 100, 10) == []


@pytest.mark.parametrize(("size", "overlap"), [(100, 50), (100, -1), (0, 0)])
def test_split_text_rejects_bad_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("x" * 500, size, overlap)


def test_sample_docs_chunk_cleanly() -> None:
    chunks = load_and_chunk(DOCS_DIR, chunk_size=800, chunk_overlap=120)
    assert len({c.doc_id for c in chunks}) >= 8
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(0 < len(c.text) <= 800 for c in chunks)


def test_load_and_chunk_requires_docs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_chunk(tmp_path, chunk_size=800, chunk_overlap=120)


def test_hash_that_is_part_of_a_heading_is_kept() -> None:
    markdown = "# Guide\n\n## Learning C#\n\nBody about C#.\n\n## Closed heading ##\n\nOther body.\n"
    chunks = chunk_markdown("guide", markdown, chunk_size=200, chunk_overlap=20)
    assert [c.section for c in chunks] == ["Learning C#", "Closed heading"]


def test_load_and_chunk_accepts_utf8_bom(tmp_path) -> None:
    (tmp_path / "bom.md").write_bytes(b"\xef\xbb\xbf# Title With BOM\n\nBody text here.\n")
    chunks = load_and_chunk(tmp_path, chunk_size=200, chunk_overlap=20)
    assert chunks[0].title == "Title With BOM"
    assert "# Title" not in chunks[0].text
