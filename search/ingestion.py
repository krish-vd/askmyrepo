"""Clone, walk, and AST-chunk a repo into CodeChunk rows.

Pipeline: clone_repo() -> iter_source_files() -> chunk_file() per file ->
build_call_graph() over all chunks in the repo. Called synchronously from
the ingest view (see views.py) — no task queue, see note there.
"""
import re
import shutil
import stat
from pathlib import Path

import git
from django.conf import settings
from tree_sitter import Language, Parser

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript

from .models import CodeChunk, Repo

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "venv", ".venv", "env",
    "__pycache__", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", "migrations",
}
MAX_FILE_SIZE = 500 * 1024  # 500KB — larger files are usually generated/data, not hand-written code

JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
SUPPORTED_AST_EXTENSIONS = {".py"} | JS_EXTENSIONS

FALLBACK_CHUNK_LINES = 60  # line-count window for non-AST fallback chunking

PY_DEF_NODE_TYPES = {"function_definition", "class_definition"}
JS_DEF_NODE_TYPES = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "generator_function_declaration",
}


def clone_repo(repo):
    """Shallow-clone repo.url into REPO_CLONE_ROOT/<repo.id>. Wipes any prior clone dir."""
    dest = settings.REPO_CLONE_ROOT / str(repo.id)
    if dest.exists():
        shutil.rmtree(dest, onerror=_force_remove_readonly)
    dest.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(repo.url, dest, depth=1)
    return dest


def _force_remove_readonly(func, path, exc_info):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def _is_probably_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


def iter_source_files(root: Path, max_files: int):
    """Yield source file paths under root, skipping vendor/binary/oversized files, capped at max_files."""
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        if _is_probably_binary(path):
            continue
        count += 1
        yield path


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _py_symbol_name(node, source_bytes: bytes):
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source_bytes)
    return None


def _js_symbol_name(node, source_bytes: bytes):
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return _node_text(child, source_bytes)
    return None


def _chunk_python_ast(source_bytes: bytes, file_path: str):
    """Walk the Python parse tree, extracting function/class defs (including nested/methods) as chunks."""
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source_bytes)

    chunks = []

    def visit(node, in_class):
        if node.type in PY_DEF_NODE_TYPES:
            name = _py_symbol_name(node, source_bytes)
            if node.type == "class_definition":
                symbol_type = CodeChunk.SymbolType.CLASS
            elif in_class:
                symbol_type = CodeChunk.SymbolType.METHOD
            else:
                symbol_type = CodeChunk.SymbolType.FUNCTION
            chunks.append({
                "symbol_name": name,
                "symbol_type": symbol_type,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "source_code": _node_text(node, source_bytes),
            })
            # Recurse into class bodies to pick up methods; a class chunk already
            # captures its full span (including methods) so we don't recurse into
            # function bodies to avoid emitting nested-function duplicates of the
            # same lines twice.
            if node.type == "class_definition":
                for child in node.children:
                    visit(child, in_class=True)
            return
        for child in node.children:
            visit(child, in_class=in_class)

    visit(tree.root_node, in_class=False)
    return chunks


def _chunk_js_ast(source_bytes: bytes, file_path: str):
    """Best-effort JS/TS chunking. TS-specific syntax (generics, decorators) may
    fail to parse cleanly since we use the JS grammar, not a dedicated TS grammar —
    acceptable per spec, falls through to line-window chunking on parse failure."""
    parser = Parser(JS_LANGUAGE)
    tree = parser.parse(source_bytes)

    chunks = []

    def visit(node, in_class):
        if node.type in JS_DEF_NODE_TYPES:
            name = _js_symbol_name(node, source_bytes)
            if node.type == "class_declaration":
                symbol_type = CodeChunk.SymbolType.CLASS
            elif node.type == "method_definition":
                symbol_type = CodeChunk.SymbolType.METHOD
            else:
                symbol_type = CodeChunk.SymbolType.FUNCTION
            chunks.append({
                "symbol_name": name,
                "symbol_type": symbol_type,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "source_code": _node_text(node, source_bytes),
            })
            if node.type == "class_declaration":
                for child in node.children:
                    visit(child, in_class=True)
            return
        # Arrow functions assigned to a const/let are common enough in JS to be
        # worth catching via the parent variable_declarator, otherwise we'd miss
        # most idiomatic JS ("const foo = () => {...}") entirely.
        if node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression"):
                name_node = node.child_by_field_name("name")
                name = _node_text(name_node, source_bytes) if name_node else None
                chunks.append({
                    "symbol_name": name,
                    "symbol_type": CodeChunk.SymbolType.FUNCTION,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "source_code": _node_text(node, source_bytes),
                })
                return
        for child in node.children:
            visit(child, in_class=in_class)

    visit(tree.root_node, in_class=False)
    return chunks


def _chunk_by_lines(text: str, window: int = FALLBACK_CHUNK_LINES):
    """Fallback for non-AST-supported languages or files that fail to parse.
    Not as semantically clean as AST chunking (may split a function mid-body),
    but guarantees every file contributes searchable context rather than being
    skipped outright."""
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), window):
        block = lines[i:i + window]
        if not any(line.strip() for line in block):
            continue
        chunks.append({
            "symbol_name": None,
            "symbol_type": CodeChunk.SymbolType.MODULE,
            "start_line": i + 1,
            "end_line": i + len(block),
            "source_code": "\n".join(block),
        })
    return chunks


def chunk_file(path: Path, repo_root: Path):
    """Return a list of chunk dicts for a single file, using AST chunking where
    supported and falling back to line-window chunking otherwise."""
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    rel_path = str(path.relative_to(repo_root))
    ext = path.suffix.lower()

    ast_chunks = None
    if ext == ".py":
        try:
            ast_chunks = _chunk_python_ast(raw, rel_path)
        except Exception:
            ast_chunks = None
    elif ext in JS_EXTENSIONS:
        try:
            ast_chunks = _chunk_js_ast(raw, rel_path)
        except Exception:
            ast_chunks = None

    if ast_chunks:
        return [{**c, "file_path": rel_path} for c in ast_chunks]

    # No top-level defs found (e.g. a script with no functions) or AST parsing
    # unavailable/failed for this file type -> line-window fallback.
    return [{**c, "file_path": rel_path} for c in _chunk_by_lines(text)]


CALL_PATTERN_CACHE_LIMIT = 2000  # guard against pathological repos with huge symbol counts


def build_call_graph(repo):
    """Best-effort call-graph pass: for every named chunk, regex-search other
    chunks' source for `<name>(` to approximate "calls". This is a heuristic,
    not real static analysis — it will produce false positives (e.g. a method
    named the same as an unrelated function in another file) and false
    negatives (aliased imports, dynamic dispatch). Good enough to power
    "show me direct callers/callees" in the retrieval trace without building
    a real call-graph analyzer, which is out of scope for this project.
    """
    chunks = list(repo.chunks.exclude(symbol_name__isnull=True).exclude(symbol_name=""))
    if not chunks or len(chunks) > CALL_PATTERN_CACHE_LIMIT:
        return

    patterns = {c.id: re.compile(re.escape(c.symbol_name) + r"\s*\(") for c in chunks}

    for chunk in chunks:
        callees = []
        for other in chunks:
            if other.id == chunk.id:
                continue
            if patterns[other.id].search(chunk.source_code):
                callees.append(other.id)
        if callees:
            chunk.calls.add(*callees)


def ingest_repo(repo):
    """Orchestrates the full pipeline for a Repo: clone -> chunk -> embed ->
    call-graph. Runs synchronously in the request/management-command that
    calls it. A production version would hand this off to a task queue
    (Celery/RQ) so the HTTP request returns immediately; out of scope here —
    MAX_INGEST_FILES keeps a single run bounded instead.
    """
    from .embedding import embed_chunks

    try:
        repo.status = Repo.Status.CLONING
        repo.save(update_fields=["status"])
        repo_dir = clone_repo(repo)

        if not repo.name:
            repo.name = repo.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

        repo.status = Repo.Status.CHUNKING
        repo.save(update_fields=["status", "name"])

        chunk_objs = []
        for path in iter_source_files(repo_dir, settings.MAX_INGEST_FILES):
            for c in chunk_file(path, repo_dir):
                chunk_objs.append(CodeChunk(
                    repo=repo,
                    file_path=c["file_path"],
                    symbol_name=c["symbol_name"],
                    symbol_type=c["symbol_type"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    source_code=c["source_code"],
                ))
        CodeChunk.objects.bulk_create(chunk_objs, batch_size=200)

        repo.status = Repo.Status.EMBEDDING
        repo.save(update_fields=["status"])

        all_chunks = list(repo.chunks.all())
        if all_chunks:
            embed_chunks(all_chunks)

        build_call_graph(repo)

        repo.status = Repo.Status.READY
        repo.save(update_fields=["status"])
    except Exception as exc:
        repo.status = Repo.Status.FAILED
        repo.error_message = str(exc)[:2000]
        repo.save(update_fields=["status", "error_message"])
        raise
