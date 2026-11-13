"""Provider-specific code stays in integrations/: core modules only reach adapters via the factory."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "helpdesk_rag"
PROVIDER_PACKAGES = {"anthropic", "ollama", "fastembed", "onnxruntime"}


def _imported_modules(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "helpdesk_rag.integrations":
                yield from (f"helpdesk_rag.integrations.{alias.name}" for alias in node.names)
            else:
                yield node.module


def test_core_modules_use_providers_only_through_the_factory() -> None:
    core_files = [p for p in SRC.rglob("*.py") if "integrations" not in p.relative_to(SRC).parts]
    assert core_files
    violations = [
        f"{path.relative_to(SRC)}: {module}"
        for path in core_files
        for module in _imported_modules(path)
        if module.split(".")[0] in PROVIDER_PACKAGES
        or (module.startswith("helpdesk_rag.integrations") and module != "helpdesk_rag.integrations.factory")
    ]
    assert violations == []
